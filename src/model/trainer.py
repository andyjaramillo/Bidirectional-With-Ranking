import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import tqdm
from tqdm import tqdm
from omegaconf import OmegaConf
from src.algos import ALGO_REGISTRY
from src.model.nn import NN, SmallerCNN
from src.utils.getData import get_data, decode_path
from src.model.replay_buffer import ReplayBuffer

class Trainer:
    def __init__(self,config):
        self.config = config
        self.Algo = ALGO_REGISTRY[config.algo]
        self.path_file = open(config.path_file, "r")  
        print("--- LOADING DATA ---")
        self.states = get_data(max_subset=10)

        print("--- DATA LOADED ----")
        self.dataset = self.path_file.readlines()[:10]
        self.test = self.dataset[-(int(0.1*len(self.dataset))):]
        self.dataset = self.dataset[:len(self.dataset) - len(self.test)]
        generator1 = torch.Generator().manual_seed(42)
        self.train_, self.val = random_split(self.dataset, [0.8, 0.2], generator=generator1)
        self.model = SmallerCNN(10)
        self.epochs = config.epochs
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.criterion, self.optimizer = self.model.initialize_cr_opt(config)
        self.search = self.Algo(self.states[0], nn=self.model)
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
            nn_value = self.model(self.search.game.decodeMap(encoded_state), self.search.game.target, self.search.game.goal_map)
            optimal_value = self.search.game.evaluateBoard((self.search.game.decodeMap(encoded_state)))
            nn_costs.append(nn_value)
            optimal_costs.append(optimal_value)
        nn_costs_tensor = torch.stack(nn_costs).to(self.device)
        nn_costs_tensor = nn_costs_tensor.squeeze()
        optimal_costs_tensor = torch.tensor(optimal_costs, requires_grad=True, device=self.device, dtype=torch.float32)
        return nn_costs_tensor, optimal_costs_tensor
    

    def create_pairwise_cost_paths(self, path, is_train=True):
        decoded_path = decode_path(path, self.search.game.decodeMap)
        self.buffer.add_pairs_from_path(decoded_path=decoded_path, target=self.search.game.target)

        sample = self.buffer.sample(64)
        
        preds = []
        targs = []
        for state, target, goal_map, cost_to_go in sample:
            preds.append(self.model(state, target, goal_map))
            targs.append(cost_to_go)
        nn_costs_tensor = torch.stack(preds).squeeze().to(self.device)
        optimal_costs_tensor = torch.tensor(targs, requires_grad=is_train, device=self.device, dtype=torch.float32)

        return nn_costs_tensor, optimal_costs_tensor



    def train(self):
        for epoch in tqdm(range(0, self.epochs), desc="Epochs"):
            for index, line in enumerate(tqdm(self.train_, desc="train states")):
                path = line.strip().split(",")
                state = self.states[index]
                self.search.reinit(state)
                
                for _ in range(self.times_to_sample):
                    self.optimizer.zero_grad()
                    nn_costs_tensor, optimal_costs_tensor = self.create_pairwise_cost_paths(path=path)
                    loss = self.criterion(nn_costs_tensor, optimal_costs_tensor)
                    correct = ((nn_costs_tensor == optimal_costs_tensor).float().sum()) / len(nn_costs_tensor)
                    self.writer.add_scalar('Loss/train', loss.item(), epoch)
                    self.writer.add_scalar('Accuracy/train', correct.item(), epoch)
                    loss.backward()
                    self.optimizer.step()
            with torch.no_grad():
                for index, line in enumerate(tqdm(self.val, desc="val states")):
                    path = line.strip().split(",")
                    state = self.states[index]
                    self.search.reinit(state)
                    for _ in range(self.times_to_sample):
                        self.optimizer.zero_grad()
                        nn_costs_tensor, optimal_costs_tensor = self.create_pairwise_cost_paths(path=path, is_train=False)
                        loss = self.criterion(nn_costs_tensor, optimal_costs_tensor)
                    correct = ((nn_costs_tensor == optimal_costs_tensor).float().sum()) / len(nn_costs_tensor)
                    self.writer.add_scalar('Loss/validation', loss.item(), epoch)
                    self.writer.add_scalar('Accuracy/train', correct.item(), epoch)
            
            torch.save(self.model.state_dict(),f'runs/experiment_{self.config.algo}/last.pt')
       
                        

        self.writer.close()
                    
    
