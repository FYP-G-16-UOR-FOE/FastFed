import argparse
import os

import torch


class ParseArgument:

    # === DEFAULT CONSTANTS ===
    FL_ROUNDS = 150
    LOCAL_EPOCHS = 4
    NUM_CLIENTS = 20
    NUM_SELECTED_CLIENTS = 10
    DATASET_PARTITIONER_MAX_CLASS_PER_CLIENT = 10
    DATASET_PARTITIONER_SEED = 42
    IS_USE_FULL_DATASET = False
    
    SELECTED_MODEL = "VGG"  # "VGG" or "NormalCNN"
    SELECTED_ALGORITHM = "FedAvg"  
    FEDPROX_MU = 0.0
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    RESULTS_SAVE_PATH = '/content/drive/MyDrive/Final_Implementations/FL_Experiments'
    RESULTS_DIR = os.path.join(RESULTS_SAVE_PATH, "FL_Results")
    
    BATCH_SIZE = 64
    
    SERVER_HOST = 'localhost'
    SERVER_PORT = 50051
    CLIENT_BASE_PORT = 60000

    SEED = 42
    WANDB_PROJECT = "FL-Test_gRPC_1"
    WANDB_KEY = "cca506f824e9db910b7b4a407afc0b36ba655c28"

    def __init__(self):
        self.args = self.parse_arguments()

    @classmethod
    def parse_arguments(cls):
        """Parse command-line arguments using CAPITALIZED class defaults."""
        parser = argparse.ArgumentParser(description="Federated Learning gRPC Server+Client")

        # Model & Algorithm configs
        parser.add_argument("--model", type=str, default=cls.SELECTED_MODEL, 
                            choices=["VGG", "NormalCNN"], help="Model type")
        
        parser.add_argument("--algorithm", type=str, default=cls.SELECTED_ALGORITHM,
                            choices=["FedAvg", "(1-JSD)", "ClientSize(1-JSD)", "AccuracyBased", 
                                    "AccuracyBased(1-JSD)", "SEBW", "AccuracyBased_SEBW", "FedProx", "CAFA"],
                            help="FL algorithm")
        
        # FL parameters
        parser.add_argument("--fl-rounds", type=int, default=cls.FL_ROUNDS, help="Number of FL rounds")
        parser.add_argument("--local-epochs", type=int, default=cls.LOCAL_EPOCHS, help="Local training epochs")
        parser.add_argument("--num-clients", type=int, default=cls.NUM_CLIENTS, help="Total number of clients")
        parser.add_argument("--num-selected-clients", type=int, default=cls.NUM_SELECTED_CLIENTS, help="Clients selected per round")
        parser.add_argument("--batch-size", type=int, default=cls.BATCH_SIZE, help="Batch size")
        parser.add_argument("--fedprox-mu", type=float, default=cls.FEDPROX_MU, help="FedProx mu parameter")
        parser.add_argument("--max-class-per-client", type=int, 
                            default=cls.DATASET_PARTITIONER_MAX_CLASS_PER_CLIENT, 
                            help="Max classes per client for non-IID distribution")

        # Dataset & Device
        parser.add_argument("--use-full-dataset", action="store_true", default=cls.IS_USE_FULL_DATASET,
                            help="Use full train dataset in each client")
        parser.add_argument("--device", type=str, default=str(cls.DEVICE), 
                            choices=["cuda", "cpu"], help="Device to use")

        # Server/Client configs
        parser.add_argument("--server-host", type=str, default=cls.SERVER_HOST, help="Server hostname")
        parser.add_argument("--server-port", type=int, default=cls.SERVER_PORT, help="Server port")
        parser.add_argument("--client-base-port", type=int, default=cls.CLIENT_BASE_PORT, help="Client base port")

        # Results & Logging
        parser.add_argument("--results-dir", type=str, default=cls.RESULTS_DIR, help="Results directory")
        parser.add_argument("--wandb-project", type=str, default=cls.WANDB_PROJECT, help="W&B project name")
        parser.add_argument("--wandb-key", type=str, default=cls.WANDB_KEY, help="W&B API key")
        parser.add_argument("--seed", type=int, default=cls.SEED, help="Random seed")

        return parser.parse_args()