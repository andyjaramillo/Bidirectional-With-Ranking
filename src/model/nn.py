import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import lightning as L
import os
import csv
from torchmetrics import Accuracy
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
    def initialize_cr_opt(self, config):
        """
        Initialize loss criterion and optimizer.
        
        Returns:
            tuple: (criterion, optimizer)
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        return criterion, optimizer
    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
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
        find_box_pos = np.where(x3d == 4)
        pos=list(zip(*(find_box_pos))) ## (3,2) list of positions of the boxes
        x1d = x3d.ravel() ## (100,) flatten the 3d tensor to 1d
        
        y1d = self.to_categorical( x1d, 5 ) ## (100, 5 )
        
        y4d = y1d.reshape( [dim1, dim2, 5]) ## (10, 10, 5)
   
        for i in range(len(pos)):
            y4d[Tar[i][0]][Tar[i][1]][2] = 1
            y4d[pos[i][0]][pos[i][1]][2] = 1
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
        box_on_T=[]

        current_box_locations =  list(zip(*(np.where(state == 4))))

        # for i in range(len(box_tar)):
        #     ## where the boxes are
        #     if state[box_tar[i][0]][box_tar[i][1]] == 4:
        #         box_on_T.append([box_tar[i][0],box_tar[i][1]])
        ## if completed
        if len(box_on_T) == len(box_tar):
            return 0

        old_state = state   ## (10, 10) game
  
        categorical_state = self.to_categorical_tensor(old_state,box_tar,10,10) ## (10, 10, 5) categorical state
        # print(categorical_state)
        categorical_goal_state = self.to_categorical_tensor(goal_state, box_tar, 10, 10)
        
        return categorical_state.reshape(1,10,10,5), categorical_goal_state.reshape(1,10,10,5)



class NNLightning(L.LightningModule):
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
        state, path = batch
        
        # output = self(inputA, box_tar,inputB)
        # loss = self.criterion(output.squeeze(), target.float())
        # self.log('train_loss', loss)
        # self.writer.writerow([self.current_epoch, loss.item()])
        # return loss

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