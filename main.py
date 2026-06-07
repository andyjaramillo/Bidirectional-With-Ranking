from src.model.trainer import Trainer
import argparse

from omegaconf import OmegaConf

def parse_arguments():
    """
    Sets up and parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Sokoban Solver - Bidirectional A* Search with Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Search Settings ---
    search_group = parser.add_argument_group("Search Algorithm Settings")

    search_group.add_argument(
        '--config', 
        dest='config',
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration file (YAML or JSON)"
    )

    search_group.add_argument(
        '--ckpt_path', 
        dest='ckpt_path',
        type=str,
        default="",
        help="Path to the .pt weights"
    )


    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)
    args = parse_arguments()
    config = OmegaConf.load(args.config)
    trainer = Trainer(config) 
    if args.ckpt_path != "":
        trainer.load_checkpoint(args.ckpt_path)
        trainer.test()
    else:
        trainer.train()   
