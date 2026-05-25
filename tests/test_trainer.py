from src.model.trainer import Trainer
from omegaconf import OmegaConf

config = {
    "epochs": 10,
"learning_rate": 0.001,
"algo": "anchor",
"batch_size": 1,
"path_file": "data/paths.txt",
}

class TestTrainer:
    trainer: Trainer
    def trainer_initialization(self):
        configs = OmegaConf.create(config)
        trainer = Trainer(config=configs)
        self.trainer = trainer

    def test_create_cost_paths(self):
        self.trainer_initialization()
        for line in self.trainer.train_:
            path = line.strip().split(",")
            nn_costs_tensor, optimal_costs_tensor = self.trainer.create_cost_paths(path=path)
            assert nn_costs_tensor.shape == optimal_costs_tensor.shape
