import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import os
import csv
try:
    import lightning as L
    from torchmetrics import Accuracy
    _LIGHTNING_AVAILABLE = True
except Exception:
    L = None
    _LIGHTNING_AVAILABLE = False

# Cached identity matrix used by NN.to_categorical for one-hot encoding
# (avoids re-allocating per forward pass).
_EYE5 = np.eye(5, dtype='uint8')
# --- 1. Positional Encoding Layer ---

def get_positional_encoding(H, W, C):
    """
    Calculates the 2D positional encoding tensor.
    
    Args:
        H (int): Height.
        W (int): Width.
        C (int): Number of channels.
        
    Returns:
        torch.Tensor: Positional encoding signal.
    """
    """
    Calculates the 2D positional encoding tensor (1, C, H, W) based on the
    original 'Attention Is All You Need' formulation, but adapted for 2D.
    The signal combines H-encoding for the first C/2 channels and W-encoding
    for the second C/2 channels, as suggested by the Keras code structure.
    """
    if C % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with C % 4 != 0 (got C={:d})".format(C))

    # C is the total number of channels (180 in the user's model)
    # The Keras implementation calculates div_term based on C/2, as each
    # dimension (H and W) uses C/2 channels, split into sin/cos pairs.
    half_C = C // 2

    # div_term: 1 / (10000^(2i/d_model)) where d_model = half_C
    i = torch.arange(0, half_C, 2, dtype=torch.float32)
    div_term = torch.exp(i * (-math.log(10000.0) / half_C))

    # 1D positions
    pos_w = torch.arange(0, W, dtype=torch.float32).unsqueeze(1) # W x 1
    pos_h = torch.arange(0, H, dtype=torch.float32).unsqueeze(1) # H x 1

    # Initialize the PE tensor in (H, W, C) format
    pe = torch.zeros((H, W, C), dtype=torch.float32)

    # Calculate 1D encodings
    sin_h = torch.sin(pos_h * div_term) # (H, C/4)
    cos_h = torch.cos(pos_h * div_term) # (H, C/4)
    sin_w = torch.sin(pos_w * div_term) # (W, C/4)
    cos_w = torch.cos(pos_w * div_term) # (W, C/4)

    # 1. H Encoding (first half_C channels)
    # Replicate H-encoding across W dimension (H, W, C/4)
    pe_h_sin = sin_h.unsqueeze(1).repeat(1, W, 1)
    pe_h_cos = cos_h.unsqueeze(1).repeat(1, W, 1)
    # Interleave sin and cos
    pe[:, :, 0:half_C:2] = pe_h_sin
    pe[:, :, 1:half_C:2] = pe_h_cos

    # 2. W Encoding (second half_C channels)
    # Replicate W-encoding across H dimension (H, W, C/4)
    pe_w_sin = sin_w.unsqueeze(0).repeat(H, 1, 1)
    pe_w_cos = cos_w.unsqueeze(0).repeat(H, 1, 1)
    # Interleave sin and cos
    pe[:, :, half_C::2] = pe_w_sin
    pe[:, :, half_C + 1::2] = pe_w_cos

    # Permute H, W, C to PyTorch's C, H, W and add batch dim (1, C, H, W)
    return pe.permute(2, 0, 1).unsqueeze(0)


class AddPositionalEncoding(nn.Module):
    """
    Module to add positional encoding to input features.
    """
    def __init__(self, dim):
        """
        Initialize the positional encoding layer.
        
        Args:
            dim (int): Input dimension.
        """
        super().__init__()
        self.dim = dim
        self.register_buffer('signal', None) # Stores the signal buffer

    def forward(self, inputs):
        # inputs shape: (N, C, H, W)
        _, C, H, W = inputs.shape

        # Dynamically generate or resize PE signal if needed
        # This mimics the Keras 'build' phase for dynamic input shapes
        if self.signal is None or self.signal.shape[2] != H or self.signal.shape[3] != W or self.signal.shape[1] != C:
            pe = get_positional_encoding(H, W, C)
            self.register_buffer('signal', pe)

        # The signal tensor is (1, C, H, W). PyTorch's broadcasting adds it correctly.
        return inputs + self.signal


# --- 2. Attention Augmentation Layer ---

class AttentionAugmentation2D(nn.Module):
    """
    Implementation of the Attention Augmentation Layer.
    Performs multi-head self-attention across spatial dimensions.
    """
    """
    Conceptual PyTorch implementation of the Attention Augmentation Layer.
    It performs multi-head self-attention across the spatial dimensions (H*W)
    and projects the output back to the original channel dimension.
    
    Args based on user's Keras code: (180, 60, 60, 2)
    in_channels=180, qk_dim=60, v_dim=60, num_heads=2
    """
    def __init__(self, in_channels, qk_dim, v_dim, num_heads):
        """
        Initialize the AttentionAugmentation2D layer.
        
        Args:
            in_channels (int): Number of input channels.
            qk_dim (int): Query/Key dimension.
            v_dim (int): Value dimension.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.in_channels = in_channels
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_heads = num_heads

        # Projections for Q, K, V (Conv1d applied spatially)
        self.proj_q = nn.Conv2d(in_channels, qk_dim * num_heads, kernel_size=1)
        self.proj_k = nn.Conv2d(in_channels, qk_dim * num_heads, kernel_size=1)
        self.proj_v = nn.Conv2d(in_channels, v_dim * num_heads, kernel_size=1)

        # Output Projection: combines the heads and projects back to in_channels
        self.output_proj = nn.Conv2d(num_heads * v_dim, in_channels, kernel_size=1)

        # Scaling factor for attention score (sqrt(d_k))
        self.scale = qk_dim ** -0.5

    def forward(self, x):
        # x shape: (N, C, H, W)
        N, C, H, W = x.shape

        # 1. Project Q, K, V
        # Reshape for attention: (N, Hn * d_k/d_v, H, W) -> (N, Hn, H*W, d_k/d_v)
        Q = self.proj_q(x).view(N, self.num_heads, self.qk_dim, H * W).permute(0, 1, 3, 2) # (N, Hn, HW, d_k)
        K = self.proj_k(x).view(N, self.num_heads, self.qk_dim, H * W)                     # (N, Hn, d_k, HW)
        V = self.proj_v(x).view(N, self.num_heads, self.v_dim, H * W).permute(0, 1, 3, 2)  # (N, Hn, HW, d_v)

        # 2. Attention: QK^T / sqrt(d_k)
        QK = torch.matmul(Q, K) * self.scale # (N, Hn, HW, HW)

        # 3. Softmax
        A = F.softmax(QK, dim=-1)

        # 4. Apply Attention to V: A * V
        # (N, Hn, HW, HW) @ (N, Hn, HW, d_v) -> (N, Hn, HW, d_v)
        attention_output = torch.matmul(A, V)

        # 5. Reshape and Project
        # Permute to (N, Hn, d_v, HW) -> concatenate heads -> (N, Hn * d_v, H*W)
        attention_output = attention_output.permute(0, 1, 3, 2).reshape(N, self.num_heads * self.v_dim, H, W)

        # Project back to in_channels (180)
        output = self.output_proj(attention_output)
        
        return output


# --- 3. Main Neural Network Model ---

class NN(nn.Module):
    """
    Multi-input neural network with attention mechanism for heuristic evaluation.
    """
    def __init__(self, dim):
        """
        Initialize the NN model.
        
        Args:
            dim (int): Input dimension.
        """
        super().__init__()
        self.dim = dim
        self.relu = nn.ReLU()
        
        # --- Shared Convolutional Stem ---
        # Initial input channels: 5 (InputA) + 5 (InputB) = 10
        self.conv1 = nn.Conv2d(10, 64, kernel_size=3, padding=1)
        
        # Input channels for conv2 to conv7 is always (64 + 10 = 74)
        # Output channels for conv2 to conv7 is always 64
        self.conv_blocks = nn.ModuleList([
            nn.Conv2d(74, 64, kernel_size=3, padding=1) for _ in range(2, 8)
        ])

        # --- Head 1 Modules (op1) ---
        # Input to conv8-1 is h1 (74 channels)
        self.conv8_1 = nn.Conv2d(74, 180, kernel_size=3, padding=1)
        
        # Channels 180, qk_dim 60, v_dim 60, heads 2
        self.att8 = AttentionAugmentation2D(180, 60, 60, 2)
        self.pe8 = AddPositionalEncoding(self.dim) # dim = 180
        # Input to conv9-1 is (180 + 180 + 10 = 370 channels)
        self.conv9_1 = nn.Conv2d(370, 180, kernel_size=3, padding=1)
        self.att9 = AttentionAugmentation2D(180, 60, 60, 2)
        self.pe9 = AddPositionalEncoding(self.dim)
        
        self.conv10_1 = nn.Conv2d(370, 180, kernel_size=3, padding=1)
        self.att10 = AttentionAugmentation2D(180, 60, 60, 2)
        self.pe10 = AddPositionalEncoding(self.dim)
        
        self.conv11_1 = nn.Conv2d(370, 180, kernel_size=3, padding=1)
        self.att11 = AttentionAugmentation2D(180, 60, 60, 2)
        self.pe11 = AddPositionalEncoding(self.dim)

        # Head 1 Classifier (Input 370 channels from last concat)
        self.avg_pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = nn.Linear(370, 256)
        self.op1 = nn.Linear(256, 8) # 8 classes

        # --- Head 2 Modules (op2) ---
        self.conv8_2 = nn.Conv2d(74, 180, kernel_size=3, padding=1)
        self.att15 = AttentionAugmentation2D(180, 60, 60, 2)
        self.pe15 = AddPositionalEncoding(self.dim)
        
        self.conv9_2 = nn.Conv2d(370, 180, kernel_size=3, padding=1)
        self.att16 = AttentionAugmentation2D(180, 60, 60, 2)
        self.pe16 = AddPositionalEncoding(self.dim)
        
        self.conv10_2 = nn.Conv2d(370, 180, kernel_size=3, padding=1)
        self.att17 = AttentionAugmentation2D(180, 60, 60, 2)
        self.pe17 = AddPositionalEncoding(self.dim)
        
        self.conv11_2 = nn.Conv2d(370, 180, kernel_size=3, padding=1)
        self.att18 = AttentionAugmentation2D(180, 60, 60, 2)
        self.pe18 = AddPositionalEncoding(self.dim)

        # Head 2 Classifier (Input 370 channels from last concat)
        self.avg_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.dense2 = nn.Linear(370, 256)
        self.op2 = nn.Linear(256, 1) # 1 output


    def forward(self, inputA, box_tar,inputB):
        """
        Forward pass of the neural network.
        
        Args:
            inputA (torch.Tensor): First input (e.g., current state).
            inputB (torch.Tensor): Second input (e.g., goal state).
            
        Returns:
            torch.Tensor: Predicted value.
        """
        inputA, inputB = self.reshapeInputs(inputA, box_tar,inputB)
        # The Keras model assumes N, H, W, C. PyTorch uses N, C, H, W.
        # Permute inputs from (N, H, W, 5) to (N, 5, H, W)
        inputA = torch.from_numpy(inputA).permute(0, 3, 1, 2).float()
        inputB = torch.from_numpy(inputB).permute(0, 3, 1, 2).float()

        # Initial Input Combination (inp: N, 10, H, W)
        inp = torch.cat([inputA, inputB], dim=1)

        # --- Shared Stem (conv1 to conv7) ---
        # Conv1
        out = self.conv1(inp)
        b = self.relu(out) # (N, 64, H, W)
        x = torch.cat([b, inp], dim=1) # b1: (N, 74, H, W)

        # Conv2 to Conv7
        for conv in self.conv_blocks:
            x_out = self.relu(conv(x)) # (N, 64, H, W)
            x = torch.cat([x_out, inp], dim=1) # h1 / next input: (N, 74, H, W)
        
        h1 = x # Final output of the shared stem

        # # ======================================================================
        # # HEAD 1 (op1)
        # # ======================================================================
        # # Conv8-1
        # i = self.relu(self.conv8_1(h1))
        # att8 = self.att8(i)
        # pe8 = self.pe8(i)
        # x1 = torch.cat([att8, pe8, inp], dim=1) # (N, 370, H, W)

        # # Conv9-1
        # j = self.relu(self.conv9_1(x1))
        # att9 = self.att9(j)
        # pe9 = self.pe9(j)
        # x1 = torch.cat([att9, pe9, inp], dim=1)

        # # Conv10-1
        # k = self.relu(self.conv10_1(x1))
        # att10 = self.att10(k)
        # pe10 = self.pe10(k)
        # x1 = torch.cat([att10, pe10, inp], dim=1)

        # # Conv11-1
        # l = self.relu(self.conv11_1(x1))
        # att11 = self.att11(l)
        # pe11 = self.pe11(l)
        # f1 = torch.cat([att11, pe11, inp], dim=1) # (N, 370, H, W)

        # # Classifier 1
        # # Global Average Pooling and Flatten (N, 370, 1, 1) -> (N, 370)
        # f1 = self.avg_pool1(f1).flatten(1)
        # d1 = self.relu(self.dense1(f1))
        # op1 = F.softmax(self.op1(d1), dim=-1) # Output 1 with softmax (8 classes)

        # ======================================================================
        # HEAD 2 (op2)
        # ======================================================================
        # Conv8-2
        p = self.relu(self.conv8_2(h1))
        att15 = self.att15(p)
        pe15 = self.pe15(p)
        x2 = torch.cat([att15, pe15, inp], dim=1) # (N, 370, H, W)

        # Conv9-2
        q = self.relu(self.conv9_2(x2))
        att16 = self.att16(q)
        pe16 = self.pe16(q)
        x2 = torch.cat([att16, pe16, inp], dim=1)

        # Conv10-2
        r = self.relu(self.conv10_2(x2))
        att17 = self.att17(r)
        pe17 = self.pe17(r)
        x2 = torch.cat([att17, pe17, inp], dim=1)

        # Conv11-2
        s = self.relu(self.conv11_2(x2))
        att18 = self.att18(s)
        pe18 = self.pe18(s)
        f2 = torch.cat([att18, pe18, inp], dim=1) # (N, 370, H, W)

        # Classifier 2
        f2 = self.avg_pool2(f2).flatten(1)
        d2 = self.relu(self.dense2(f2))
        op2 = self.op2(d2) # Output 2 (1 output, no final activation in original Keras)

        return op2
    def initialize_cr_opt(self):
        """
        Initialize loss criterion and optimizer.
        
        Returns:
            tuple: (criterion, optimizer)
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return criterion, optimizer
    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        # Cache the identity matrix to avoid re-allocating per call.
        if num_classes == 5:
            return _EYE5[y]
        return np.eye(num_classes, dtype='uint8')[y]

    def to_categorical_tensor(self, x3d,Tar,dim1,dim2):
        """
        Convert a 3D tensor to a categorical one-hot representation.
        
        Args:
            x3d (torch.Tensor): Input tensor. (10,10)
            Tar (list): Target locations.
            dim1 (int): Height.
            dim2 (int): Width.
            
        Returns:
            torch.Tensor: Categorical tensor.
        """
        # One-hot encode, then mark channel-2 at box and target cells.
        y4d = self.to_categorical(x3d.ravel(), 5).reshape(dim1, dim2, 5)
        box_r, box_c = np.where(x3d == 4)
        y4d[box_r, box_c, 2] = 1
        for tr, tc in Tar:
            y4d[tr, tc, 2] = 1
        return y4d


    
    def reshapeInputs(self,state, box_tar,goal_state):
        """
        Perform inference for a single state.
        
        Args:
            state (np.ndarray): Current state.
            box_tar (list): Target locations. (3,2) So 3 targets with each being (x,y)
            goal_state (np.ndarray): Goal state.
            
        Returns:
            float: Predicted value.
        """
        categorical_state = self.to_categorical_tensor(state, box_tar, 10, 10)
        categorical_goal_state = self.to_categorical_tensor(goal_state, box_tar, 10, 10)
        return (categorical_state.reshape(1, 10, 10, 5),
                categorical_goal_state.reshape(1, 10, 10, 5))



class SmallCNN(nn.Module):
    """Compact CNN heuristic for 10x10 Sokoban boards.

    Input is the same (state, target, goal_state) signature as NN: two
    one-hot encoded boards of shape (10, 10, 5) which concatenate into a
    10-channel tensor. Trains fast on CPU.
    """

    def __init__(self, channels=32):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(10, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(channels, 64)
        self.fc2 = nn.Linear(64, 1)
        # Reuse NN's preprocessing helpers
        self._helper = NN.__new__(NN)  # uninitialized; only used for methods
        nn.Module.__init__(self._helper)

    def _prep(self, state, box_tar, goal_state):
        a, b = NN.reshapeInputs(self._helper, state, box_tar, goal_state)
        # Concatenate the two one-hot boards in numpy, then a single
        # from_numpy/permute/float instead of two of each plus a torch.cat.
        ab = np.concatenate([a, b], axis=-1)  # (1, 10, 10, 10)
        # .contiguous() after permute: required for correct MPS backward
        # (permute leaves a non-contiguous view), negligible cost on CPU.
        t = torch.from_numpy(ab).permute(0, 3, 1, 2).float().contiguous()
        dev = self.conv1.weight.device
        return t if dev.type == "cpu" else t.to(dev)

    def _conv_stack(self, x):
        # Functional API avoids nn.Module.__call__ dispatch (_call_impl,
        # __getattr__, _get_tracing_state) on every layer — ~9% of search time.
        x = F.relu(F.conv2d(x, self.conv1.weight, self.conv1.bias, padding=1))
        x = F.relu(F.conv2d(x, self.conv2.weight, self.conv2.bias, padding=1))
        x = F.relu(F.conv2d(x, self.conv3.weight, self.conv3.bias, padding=1))
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = F.relu(F.linear(x, self.fc1.weight, self.fc1.bias))
        return F.linear(x, self.fc2.weight, self.fc2.bias)

    def forward(self, state, box_tar, goal_state):
        return self._conv_stack(self._prep(state, box_tar, goal_state))

    def forward_batch(self, states, targets, others):
        """Vectorized forward pass over a list of (state, target, other) inputs.

        Per-sample one-hot encoding runs in NumPy, but the whole batch is
        stacked into ONE array and transferred to the model's device in a
        SINGLE host→device copy — critical for MPS, where per-sample
        transfers dominate. The conv stack then runs once on the batch.
        """
        mats = []
        for s, t, o in zip(states, targets, others):
            a, b = NN.reshapeInputs(self._helper, s, t, o)
            mats.append(np.concatenate([a, b], axis=-1))  # (1, 10, 10, 10)
        ab = np.concatenate(mats, axis=0)                 # (B, 10, 10, 10)
        x = torch.from_numpy(ab).permute(0, 3, 1, 2).float().contiguous()
        dev = self.conv1.weight.device
        if dev.type != "cpu":
            x = x.to(dev)  # single transfer for the whole batch
        return self._conv_stack(x).squeeze(-1)

    def initialize_cr_opt(self, lr=1e-3, loss_type="mae"):
        if loss_type == "mse":
            crit = nn.MSELoss()
        else:
            crit = nn.L1Loss()  # MAE — robust to heavy-tailed pair distances
        return crit, optim.Adam(self.parameters(), lr=lr)


class SmallCNNAttn(SmallCNN):
    """SmallCNN with one spatial self-attention block between the conv tower
    and the MLP head.

    The conv stack produces a (N, C, 10, 10) feature map; we treat the 100
    cells as tokens (each a C-dim channel vector), add a 2D positional
    encoding so the layer knows *where* each cell is, and run one Transformer
    encoder block (multi-head self-attention + FFN, pre-norm + residual).
    This gives every cell a GLOBAL receptive field in a single layer — a box
    cell can attend to a distant target cell — which the 7x7-RF conv stack
    plus global-average-pool cannot do. Tokens are mean-pooled into the head.

    pe="sincos" (default): parameter-free 2D sinusoidal encoding (separate H
    and W components — respects the grid). pe="learned": a learned per-cell
    embedding (one vector per grid position).

    Reuses SmallCNN's _prep/forward/forward_batch unchanged; only the core
    stack (_conv_stack) is overridden, so both the search and training paths
    route through attention automatically.
    """

    def __init__(self, channels=32, heads=4, ff_mult=2, pe="sincos"):
        super().__init__(channels)
        if channels % heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
        self.H = self.W = 10
        n_tok = self.H * self.W
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * ff_mult),
            nn.ReLU(),
            nn.Linear(channels * ff_mult, channels),
        )
        self.pe_mode = pe
        if pe == "learned":
            self.pos = nn.Parameter(torch.zeros(1, n_tok, channels))
            nn.init.trunc_normal_(self.pos, std=0.02)
        elif pe == "sincos":
            # (1, C, H, W) sinusoidal -> (1, H*W, C) token-major, as a buffer.
            pe_t = get_positional_encoding(self.H, self.W, channels)
            self.register_buffer("pos", pe_t.flatten(2).permute(0, 2, 1).contiguous())
        else:
            raise ValueError(f"unknown pe mode: {pe!r}")

    def _conv_stack(self, x):
        # Conv tower (same as SmallCNN, but stop before pooling).
        x = F.relu(F.conv2d(x, self.conv1.weight, self.conv1.bias, padding=1))
        x = F.relu(F.conv2d(x, self.conv2.weight, self.conv2.bias, padding=1))
        x = F.relu(F.conv2d(x, self.conv3.weight, self.conv3.bias, padding=1))
        N, C, H, W = x.shape                       # (N, C, 10, 10)
        tok = x.flatten(2).permute(0, 2, 1)        # (N, HW, C) tokens
        tok = tok + self.pos                       # 2D positional encoding
        # Pre-norm Transformer encoder block.
        a = self.norm1(tok)
        a, _ = self.attn(a, a, a, need_weights=False)
        tok = tok + a
        tok = tok + self.ff(self.norm2(tok))
        pooled = tok.mean(dim=1)                    # (N, C) token mean-pool
        h = F.relu(F.linear(pooled, self.fc1.weight, self.fc1.bias))
        return F.linear(h, self.fc2.weight, self.fc2.bias)


def build_model(name="smallcnn", channels=32, **kw):
    """Factory used by online_run to select a heuristic model by name."""
    name = name.lower()
    if name in ("smallcnn", "cnn", "small"):
        return SmallCNN(channels)
    if name in ("smallcnn_attn", "attn", "cnn_attn"):
        return SmallCNNAttn(channels, **kw)
    raise ValueError(f"unknown model: {name!r}")


_LightningBase = L.LightningModule if _LIGHTNING_AVAILABLE else object


class NNLightning(_LightningBase):
    """
    PyTorch Lightning wrapper for the NN model.
    """
    def __init__(self, dim):
        super().__init__()
        self.model = NN(dim)
        self.criterion, self.optimizer = self.model.initialize_cr_opt()
        log_dir = "./loss_logs"
        file_count = len([f for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f))])
        new_file_num = file_count + 1
        filename = f"{new_file_num}.txt"
        filepath = os.path.join(log_dir, filename)
        headers = ['epoch', 'step', 'nn_cost', 'optimal_cost', 'loss']
        self.f = open(filepath, mode='w', newline='')
        self.writer = csv.writer(self.f)
        self.writer.writerow(headers)
    def forward(self, inputA, box_tar,inputB):
        return self.model(inputA, box_tar,inputB)

    def training_step(self, batch, batch_idx):
        inputA, box_tar,inputB, target = batch
        output = self(inputA, box_tar,inputB)
        loss = self.criterion(output.squeeze(), target.float())
        self.log('train_loss', loss)
        self.writer.writerow([self.current_epoch, loss.item()])
        return loss

    def validation_step(self, batch, batch_idx):
        inputA, box_tar,inputB, target = batch
        output = self(inputA, box_tar,inputB)
        loss = self.criterion(output.squeeze(), target.float())
        self.log('val_loss', loss)
        return loss
    def configure_optimizers(self):
        return self.optimizer
    
    def on_fit_end(self, trainer, pl_module):
        self.f.close()