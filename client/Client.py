import copy
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import jensenshannon  # JS calculation
from scipy.stats import wasserstein_distance  # EMD calculation
from torch.utils.data import DataLoader, random_split

torch.backends.cudnn.benchmark = True

class Client:
    """Federated Learning Client class for local dataset handling, training, and evaluation."""

    def __init__(
            self, id, ip_address, port, device, model, client_dataset, batch_size,
            selected_algorithm, is_use_full_dataset, model_type,
            local_epochs, fedprox_mu
    ):
        """
        Initialize the client with dataset splits and dataloaders.

        Args:
            id (int): Unique client ID.
            client_dataset_list (list): List of datasets for all clients.
            batch_size (int): Batch size for data loaders.
        """
        self.id = id
        self.ip_address = ip_address
        self.port = port
        self.batch_size = batch_size
        self.device = device
        self.client_model = model
        self.client_model.to(self.device)  # Move model to device (CPU or CUDA)
        self.local_training_details = {
            "train_times": [],
            "train_accuracies": [],
            "val_accuracies": [],
            "train_losses": [],
            "val_losses": [],
            "iid_measure_time": None
        }
        self.client_train_config = {
            "selected_algorithm": selected_algorithm,
            "is_use_full_dataset": is_use_full_dataset,
            "model_type": model_type,
            "local_epochs": local_epochs,
            "mu": fedprox_mu
        }
        self.iid_measure = None

        full_dataset = client_dataset

        # Split dataset: 80% train, 20% validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(int(self.id)))

        # Create data loaders
        self.trainloader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        self.valloader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        self.full_dataloader = DataLoader(full_dataset, batch_size=self.batch_size, shuffle=True)

        # Save dataset sizes
        self.dataset_size = train_size
        self.val_size = val_size

    def receive_global_model(self, state_dict):
        self.client_model.load_state_dict(state_dict, strict=True)
        self.client_model.to(self.device)

    def start_client_local_training(self):
        self.train_client_model()

    def get_client_updates(self):
        return self.client_model.state_dict(), self.local_training_details["train_times"][-1]
    
    def calculate_iid_measure(self):
        self.measure_iid_nature()
        return self.iid_measure

    def evaluate_model_on_validation_data(self, state_dict, model_type):
        model = copy.deepcopy(self.client_model)
        model.load_state_dict(state_dict, strict=True)
        return self.get_weighted_val_accuracy(client_model=model, model_type=model_type)
    
    def measure_iid_nature(self):
        iid_measure_method = self.get_iid_measure_method()
        iid_measure_time_start = time.time()

        if iid_measure_method == "JSD":
            iid_measure = float(self.get_agg_weight_jsd())
        elif iid_measure_method == "SEBW":
            iid_measure = float(self.get_shannon_entropy_weight_metric())
        else:
            iid_measure = None

        iid_measure_duration = time.time() - iid_measure_time_start
        self.iid_measure = iid_measure
        self.local_training_details["iid_measure_time"] = iid_measure_duration

    def get_iid_measure_method(self):
        algorithm = self.client_train_config.get("selected_algorithm", None)
        if algorithm in ["(1-JSD)", "AccuracyBased(1-JSD)"]:
            return "JSD"
        elif algorithm in ["SEBW", "AccuracyBased_SEBW"]:
            return "SEBW"
        else:
            return None
        
    # --------------------------------------------------------------------------
    # 📊 Dataset Analysis Methods
    # --------------------------------------------------------------------------
    def get_client_dataset_size(self):
        return self.dataset_size

    def calculate_label_distribution(self):
        """Return count of each class label (0–9) in the full dataset."""
        all_labels = []
        for _, labels in self.full_dataloader:
            all_labels.extend(labels.tolist())

        label_counts = Counter(all_labels)
        for i in range(10):
            label_counts.setdefault(i, 0)
        return dict(sorted(label_counts.items()))

    def calculate_label_probability_distribution(self):
        """Return normalized class probabilities for the client."""
        counts = np.array(list(self.calculate_label_distribution().values()), dtype=np.float32)
        total = counts.sum()
        return counts / total if total > 0 else np.zeros_like(counts)

    def calculate_emd(self):
        """Compute Earth Mover’s Distance (EMD) between client and reference distribution."""
        reference_counts = np.full(10, 6000, dtype=np.float32)
        client_probs = self.calculate_label_probability_distribution()
        reference_probs = reference_counts / reference_counts.sum()
        return wasserstein_distance(np.arange(10), np.arange(10), u_weights=client_probs, v_weights=reference_probs)

    def calculate_js_divergence(self):
        """Compute Jensen–Shannon Divergence between client and reference distribution."""
        reference_counts = np.full(10, 6000, dtype=np.float32)
        client_probs = self.calculate_label_probability_distribution()
        reference_probs = reference_counts / reference_counts.sum()
        return jensenshannon(client_probs, reference_probs, base=2.0)

    def calculate_shannon_entropy(self, distribution_array):
        """Compute Shannon entropy of a given distribution."""
        distribution_array = np.array(distribution_array, dtype=np.float32)
        probs = distribution_array / (distribution_array.sum() + 1e-12)
        return -np.sum(probs * np.log2(probs + 1e-12))

    def get_shannon_entropy_weight_metric(self):
        """Calculate weight metric based on entropy and dataset size."""
        dist = np.array(list(self.calculate_label_distribution().values()), dtype=np.float32)
        entropy_val = self.calculate_shannon_entropy(dist)
        return (entropy_val ** 2) * np.log2(self.dataset_size + 1)

    def get_agg_weight_jsd(self):
        """Aggregation weight = (1 - JSD)."""
        js_value = self.calculate_js_divergence()
        return (1 - js_value)

    def get_combined_agg_weight_jsd_client_size(self):
        """Aggregation weight = (1 - JSD) × dataset_size."""
        js_value = self.calculate_js_divergence()
        return (1 - js_value) * self.dataset_size

    def get_combined_agg_weight_jsd_log_client_size(self):
        """Aggregation weight = (1 - JSD) × log2(dataset_size)."""
        js_value = self.calculate_js_divergence()
        return (1 - js_value) * np.log2(self.dataset_size + 1)

    # --------------------------------------------------------------------------
    # 🧠 Model Training & Evaluation
    # --------------------------------------------------------------------------
    def train_client_model(self):
        """
        Train the client model locally using the specified dataset and training type.
        """
        # FedProx
        if self.client_train_config["selected_algorithm"] == "FedProx":
            global_model_params = copy.deepcopy(self.client_model.state_dict())
            # Move each param tensor to device if needed
            for key in global_model_params:
                global_model_params[key] = global_model_params[key].to(self.device)
        self.client_model.train()
        train_time = time.time() 

        # Optimizer & loss
        if self.client_train_config["model_type"] == "VGG":
            optimizer = optim.SGD(self.client_model.parameters(), lr=1e-1)
            criterion = F.nll_loss
        elif self.client_train_config["model_type"] == "NormalCNN":
            optimizer = optim.SGD(self.client_model.parameters(), lr=1e-2)
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Invalid model type: {self.client_train_config["model_type"]}.")

        # Dataset loader
        if self.client_train_config["is_use_full_dataset"]:
            dataloader = self.full_dataloader
        else:
            dataloader = self.trainloader

        # Train loop
        for epoch in range(self.client_train_config["local_epochs"]):
            total_loss, correct, total = 0.0, 0, 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs = self.client_model(inputs)
                loss = criterion(outputs, labels)

                # FedProx proximal term
                if self.client_train_config["selected_algorithm"] == "FedProx":
                    prox_term = 0.0
                    for name, param in self.client_model.named_parameters():
                        global_param = global_model_params[name].to(self.device)
                        prox_term += ((param - global_param) ** 2).sum()
                    loss += (self.client_train_config["fedprox_mu"] / 2) * prox_term

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            avg_loss = total_loss / total
            accuracy = 100.0 * correct / total
            print(f"[Client {self.id}] Epoch {epoch + 1}/{self.client_train_config['local_epochs']} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")

        train_duration = time.time() - train_time
        print(f"[Client {self.id}] Local training completed in {train_duration:.2f} seconds.")
        self.local_training_details["train_times"].append(train_duration)
        self.local_training_details["train_accuracies"].append(accuracy)
        self.local_training_details["train_losses"].append(avg_loss)

    # --------------------------------------------------------------------------
    # 🧪 Testing and Validation
    # --------------------------------------------------------------------------
    def test_client_model(self, model, model_type="VGG"):
        """Evaluate the trained model on a test dataset."""
        model.eval()
        total_loss, correct, total = 0.0, 0, 0

        # Select loss function
        if model_type == "VGG":
            criterion = F.nll_loss
        elif model_type == "NormalCNN":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Invalid model type: {model_type}. Expected one of {self.model_types}.")

        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return accuracy, avg_loss

    def get_weighted_val_accuracy(self, client_model, model_type="VGG"):
        """Compute validation accuracy weighted by validation dataset size."""
        accuracy, _ = self.test_client_model(model=client_model, model_type=model_type)
        weighted_accuracy = accuracy * self.val_size
        return weighted_accuracy
