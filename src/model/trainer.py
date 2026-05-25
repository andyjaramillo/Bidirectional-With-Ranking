import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import tqdm
from tqdm import tqdm
from omegaconf import OmegaConf
from src.algos import ALGO_REGISTRY
from src.model.nn import NN
from src.utils.getData import get_data, decode_path
from src.model.replay_buffer import ReplayBuffer

class Trainer:
    def __init__(self,config):
        self.Algo = ALGO_REGISTRY[config.algo]
        self.path_file = open(config.path_file, "r")  
        self.states = get_data()
        self.dataset = self.path_file.readlines()
        generator1 = torch.Generator().manual_seed(42)
        self.train_, self.val = random_split(self.dataset, [0.8, 0.2], generator=generator1)
        self.model = NN(10)
        self.epochs = config.epochs
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.criterion, self.optimizer = self.model.initialize_cr_opt(config)
        self.front = self.Algo(self.states[0], False, nn=self.model)
        self.back = self.Algo(self.front.game.initializeBackwardPuzzle(self.states[0]), True, nn=self.model)
        self.buffer = ReplayBuffer()
        ## tensorboard logs
        self.writer = SummaryWriter(f'runs/experiment_{config.algo}')
        yaml_text = OmegaConf.to_yaml(config).replace("\n", "  \n")
        self.times_to_sample = 16 ## number of times we sample from the pairwise paths
        # 4. Log to the Text tab
        self.writer.add_text("Configuration/YAML", f"yaml\n{yaml_text}", global_step=0)

    def create_cost_paths(self, path):
        nn_costs = []
        optimal_costs = []

        for encoded_state in path:
            nn_value = self.model(self.front.game.decodeMap(encoded_state), self.front.game.target, self.front.game.goal_map)
            optimal_value = self.front.game.evaluateBoard((self.front.game.decodeMap(encoded_state)))
            nn_costs.append(nn_value)
            optimal_costs.append(optimal_value)
        nn_costs_tensor = torch.stack(nn_costs).to(self.device)
        nn_costs_tensor = nn_costs_tensor.squeeze()
        optimal_costs_tensor = torch.tensor(optimal_costs, requires_grad=True, device=self.device, dtype=torch.float32)
        return nn_costs_tensor, optimal_costs_tensor
    

    def create_pairwise_cost_paths(self, path):
        decoded_path = decode_path(path, self.front.game.decodeMap)
        self.buffer.add_pairs_from_path(decoded_path=decoded_path, target=self.front.game.target)

        sample = self.buffer.sample(64)
        
        preds = []
        targs = []
        for state, target, goal_map, cost_to_go in sample:
            preds.append(self.model(state, target, goal_map))
            targs.append(cost_to_go)
        
        nn_costs_tensor = torch.stack(preds).squeeze().to(self.device)
        optimal_costs_tensor = torch.tensor(targs).to(dtype=torch.float32).to(self.device)

        return nn_costs_tensor, optimal_costs_tensor



    def train(self):
        with tqdm(total=None, desc="Searching ", unit="states") as pbar:
            for epoch in range(0, self.epochs):
                for index, line in enumerate(self.train_):
                    path = line.strip().split(",")
                    state = self.states[index]
                    self.front = self.Algo(state, False)
                    self.back = self.Algo(self.front.game.initializeBackwardPuzzle(state), True)
                    
                    for _ in range(self.times_to_sample):
                        self.optimizer.zero_grad()
                        nn_costs_tensor, optimal_costs_tensor = self.create_pairwise_cost_paths(path=path)
                        loss = self.criterion(nn_costs_tensor, optimal_costs_tensor)
                        self.writer.add_scalar('Loss/train', loss.item(), epoch)
                        loss.backward()
                        self.optimizer.step()
                        pbar.update(1)
                with torch.no_grad():
                    for index, line in enumerate(self.val):
                        path = line.strip().split(",")
                        state = self.states[index]
                        self.front = self.Algo(state, False)
                        self.back = self.Algo(self.front.game.initializeBackwardPuzzle(state), True)
                        for _ in range(self.times_to_sample):
                            self.optimizer.zero_grad()
                            nn_costs_tensor, optimal_costs_tensor = self.create_pairwise_cost_paths(path=path)
                            loss = self.criterion(nn_costs_tensor, optimal_costs_tensor)
                            loss.backward()
                        loss = self.criterion(nn_costs_tensor, optimal_costs_tensor)
                        self.writer.add_scalar('Loss/validation', loss.item(), epoch)

        self.writer.close()
                    
    
