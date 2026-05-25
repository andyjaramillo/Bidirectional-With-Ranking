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
        '--learning', '--with_learning',
        dest='learning',
        type=str,
        default="off",
        choices=["on", "off"],
        help="Enable neural network learning"
    )

    search_group.add_argument(
        '--config', 
        dest='config',
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration file (YAML or JSON)"
    )


    search_group.add_argument(
        '--with_training',
        type=str,
        default="no",
        choices=["yes", "no"],
        help="Enable neural network training"
    )
    search_group.add_argument(
        "--front_to_front",
        type=str,
        default="no",
        choices=["yes", "no"],
        help="Use Front-to-Front learning instead of Meet-in-the-Middle"
    )
    search_group.add_argument(
        "--anchor_search",
        type=str,
        default="no",
        choices=["yes", "no"],
        help="Use Anchor Search"
    )
    search_group.add_argument(
        "--ttbs",
        type=str,
        default="no",
        choices=["yes", "no"],
        help="Use Top-to-Top Bidirectional Search (TTBS) from IJCAI 2020"
    )

    # --- Experiment & Environment ---
    exp_group = parser.add_argument_group("Experiment & Environment Settings")
    exp_group.add_argument(
        "--noise", "--with_noise",
        dest='noise',
        type=str,
        default="off",
        choices=["off", "additive", "multiplicative"],
        help="Type of noise to add to heuristic scores"
    )
    exp_group.add_argument(
        "--dummy_data", "--with_dummy_data",
        dest='dummy_data',
        type=str,
        default="No",
        choices=["Yes", "No"],
        help="Use simple puzzles from test_box.txt for development"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config = OmegaConf.load(args.config)
    trainer = Trainer(config) 
    trainer.train()   
