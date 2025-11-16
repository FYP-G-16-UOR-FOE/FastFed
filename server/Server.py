import collections
import copy
import os
import io
import pickle
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import psutil
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from torchvision import transforms

import wandb
import grpc
from gRPC import ClientgRPC_pb2
from gRPC import ClientgRPC_pb2_grpc

torch.backends.cudnn.benchmark = True

class Server:
    def __init__(self, device, global_model, model_type, fl_rounds, num_clients, num_selected_clients, local_epochs, selected_algorithm, results_save_dir):
        """
        Server class for Federated Learning aggregation and coordination.
        """
        self.global_model = global_model
        self.device = device
        self.clients = {
            "clients_ids": [],
            "client_api_urls": [],
            "selected_clients_ids": [],
            "selected_clients_models": []
        }
        self.fl_train_config = {
            "fl_rounds": fl_rounds,
            "local_epochs": local_epochs,
            "model_type": model_type,
            "num_clients": num_clients,
            "num_selected_clients": num_selected_clients,
            "results_save_dir": results_save_dir,
            "is_use_full_dataset": True,
        }
        self.fl_accuracy = {
            "selected_algorithm": selected_algorithm,
            "iid_agg_weights": [],
            "accuracy_based_agg_weights": [],
        }
        self.fl_security = {}
        self.fl_performance ={
            "selected_performance_optimization_types": []
        }
        self.history = {
            "global_accuracy": [],
            "global_loss": [],
            "fl_train_time": [],
            "round_local_training_time": [],
            "round_aggregation_time": [],
            "iid_agg_weights_calc_time": [],
            "system_usage": [],
            "round_communication_time": [],
            "round_global_model_send_time": [],
            "round_client_model_recv_time": [],
            "round_global_model_size_bytes": [],
            "global_model_size_bytes": [],
        }
        
        # Global test dataset (CIFAR-10)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        global_test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
        self.global_test_dataloader = DataLoader(global_test_dataset, batch_size=100)

    def register_client(self, client_id: int, client_api_url: str):
        if client_id in self.clients["clients_ids"]:
            print(f"Client ID {client_id} is already registered.")
            return
        print(f"Registering Client ID: {client_id}")
        self.clients["clients_ids"].append(client_id)
        self.clients["client_api_urls"].append(client_api_url)

    def receive_client_updates(self, client_id: int, client_model: dict, training_time: float, iid_measure: float):
        print(f"Receiving updates from Client ID: {client_id}")
        deserialized_model = self.deserialize_model(client_model)
        self.clients["selected_clients_models"].append(deserialized_model)
        self.history["round_local_training_time"][-1] += training_time
        self.fl_accuracy["iid_agg_weights"].append(iid_measure)
        
    # ============================================================
    # 🔹 Core Server Functions
    # ============================================================
    def get_global_model_params(self):
        """Return global model parameters."""
        return self.global_model.state_dict()
    
    def serialize_model(self, model):
        """
        Convert a PyTorch model's state_dict into a JSON-serializable dictionary.
        Tensors are converted to CPU NumPy arrays, then to Python lists.
        """
        state_dict = model.state_dict()
        serialized = {
            k: (v.detach().cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v)
            for k, v in state_dict.items()
        }
        return serialized
    
    def deserialize_model(self, serialized_state):
        """
        Convert a JSON-serializable model dictionary back into a PyTorch state_dict.
        Lists → Tensors
        Scalars → Tensors
        """
        deserialized_state = {}

        for k, v in serialized_state.items():
            # Case 1: list (weights/bias tensors)
            if isinstance(v, list):
                deserialized_state[k] = torch.tensor(v)

            # Case 2: scalar numpy values that were converted using .item()
            elif isinstance(v, (int, float)):
                deserialized_state[k] = torch.tensor(v)

            # Case 3: fallback (should not happen normally)
            else:
                deserialized_state[k] = torch.tensor(v)

        return deserialized_state

    
    def cal_size(self, obj):
        """Calculate the size of a Python object in bytes."""
        return len(pickle.dumps(obj))
    
    def fed_avg(self, weights_list):
        """
        Perform standard FedAvg aggregation on a list of parameter OrderedDicts (with NumPy arrays as values).
        Returns an OrderedDict of PyTorch tensors.
        """
        avg_params = collections.OrderedDict()
        for key in weights_list[0].keys():
            stacked = np.stack([w[key] for w in weights_list])
            avg_params[key] = torch.from_numpy(np.mean(stacked, axis=0))
        return avg_params

    def weighted_fed_avg(self, weights_list, agg_weights):
        """
        Perform weighted FedAvg aggregation on a list of parameter OrderedDicts (NumPy arrays as values) and weights.
        Returns an OrderedDict of PyTorch tensors.
        """
        total_weight = sum(agg_weights)
        avg_params = collections.OrderedDict()
        for key in weights_list[0].keys():
            stacked = np.stack([w[key] * agg_weights[i] for i, w in enumerate(weights_list)])
            avg_params[key] = torch.from_numpy(np.sum(stacked, axis=0) / total_weight)
        return avg_params

    def aggregate_models(self, client_models=None, agg_weights=None):
        """
        Aggregate client models using FedAvg or Weighted FedAvg, Flower-style (NumPy conversion).
        Accepts a list of client model state_dicts (PyTorch) in client_models.
        """
        if client_models is None:
            client_models = self.selected_clients_models
        # Convert PyTorch state_dicts to OrderedDicts of NumPy arrays
        weights_list = []
        for client_model in client_models:
            state_dict = client_model.state_dict() if hasattr(client_model, "state_dict") else client_model
            weights_np = collections.OrderedDict()
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    weights_np[k] = v.cpu().numpy()
                else:
                    weights_np[k] = np.array(v)
            weights_list.append(weights_np)

        # Select aggregation function
        if not agg_weights or (len(agg_weights) > 0 and all(abs(agg_weights[i] - agg_weights[0]) < 1e-6 for i in range(len(agg_weights)))):
            print("\n🟢 Using FedAvg aggregation.")
            aggregated_model_params = self.fed_avg(weights_list)
        else:
            print("\n🟡 Using Weighted FedAvg aggregation.")
            print("Aggregation Weights :", agg_weights)
            aggregated_model_params = self.weighted_fed_avg(weights_list, agg_weights)
        self.global_model.load_state_dict(copy.deepcopy(aggregated_model_params))

    def cal_min_max_agg_weight(self, client_iid_measure_list):
        """Convert IID measure values into normalized aggregation weights."""
        d_min, d_max = min(client_iid_measure_list), max(client_iid_measure_list)
        if d_max == d_min:
            return [1.0 for _ in client_iid_measure_list]
        agg_weights = [(1 - (val - d_min) / (d_max - d_min)) for val in client_iid_measure_list]
        total = sum(agg_weights)
        return [w / total for w in agg_weights]
    
    def test_model(self):
        """Evaluate global model on test data."""
        self.global_model.eval()
        correct, total, total_loss = 0, 0, 0.0

         # Select loss function
        if self.fl_train_config["model_type"] == "VGG":
            criterion = F.nll_loss
        elif self.fl_train_config["model_type"] == "NormalCNN":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Invalid model type: {self.model_type}.")

        with torch.no_grad():
            for images, labels in self.global_test_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return accuracy, avg_loss

    def evaluate_kmeans_silhouette(self, distribution_matrix, k_range):
        """
        Evaluate silhouette scores for a range of cluster counts.
        """
        silhouette_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0)
            labels = kmeans.fit_predict(distribution_matrix)
            score = silhouette_score(distribution_matrix, labels)
            silhouette_scores.append(score)
            print(f"k={k}, silhouette score={score:.4f}")
        return silhouette_scores

    def get_clients_classification_agg_weights(self, clients):
        """
        Determine the optimal number of client clusters (k) using silhouette score.
        """

        client_dicts = [
            client.get_from_client(data={'agg_weight_type': "cafa"}, comm_type="GET_AGG_WEIGHT")
            for client in clients
        ]
        keys = sorted(client_dicts[0].keys())
        distribution_matrix = np.array([[d[k] for k in keys] for d in client_dicts], dtype=float)

        k_range = range(2, len(clients))  # test from 2 to N-1
        silhouette_scores = self.evaluate_kmeans_silhouette(distribution_matrix, k_range)

        # Plot results
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, silhouette_scores, marker='o')
        plt.title("Silhouette Score vs Number of Clusters (k)")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.xticks(k_range)
        plt.grid(True)
        plt.show()

        # Select best k
        best_k_idx = int(np.argmax(silhouette_scores))
        best_k = list(k_range)[best_k_idx]
        print(f"\nOptimal k = {best_k}, Silhouette Score = {silhouette_scores[best_k_idx]:.4f}")

        # Cluster clients
        kmeans = KMeans(n_clusters=best_k, random_state=0)
        client_cluster_labels = kmeans.fit_predict(distribution_matrix)

        num_clients = len(client_cluster_labels)
        cluster_counts = Counter(client_cluster_labels)
        clients_clus_agg_weights = np.array([
            cluster_counts[label] / num_clients for label in client_cluster_labels
        ])

        return clients_clus_agg_weights

    def get_system_usage(self, device):
        usage = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
        }
        if "cuda" in str(device):
            usage.update({
                "gpu_mem_allocated_MB": torch.cuda.memory_allocated(0) / 1e6,
                "gpu_mem_reserved_MB": torch.cuda.memory_reserved(0) / 1e6,
            })
        return usage

    def save_results(self, results, results_save_dir):
        # Build file paths
        results_file = os.path.join(results_save_dir, "results.pkl")

        # Ensure folders exist
        os.makedirs(results_save_dir, exist_ok=True)

        # Save history in one file
        with open(results_file, "wb") as f:
            pickle.dump(results, f)

        print(f"✅ All result saved successfully.")


    def get_avg_val_accuracy_based_agg_weights(self, selected_clients_models, selected_clients):
        """
        Calculate accuracy-based aggregation weights.
        """
        accuracy_based_agg_weights = []
        for i in range(len(selected_clients)):
            client_model = selected_clients_models[i]
            client_weighted_val_accuracies = []
            for j in range(len(selected_clients)):
                if i == j:
                    continue
                test_client = selected_clients[j]

                get_weighted_val_acc_comm_data = {
                    "client_model": client_model,
                    "model_type": self.model_type,
                }
                client_weighted_val_accuracy = test_client.get_from_client(data=get_weighted_val_acc_comm_data, comm_type="GET_WEIGHTED_VALIDATION_ACCURACY")
                client_weighted_val_accuracies.append(client_weighted_val_accuracy)
            average_client_weighted_val_accuracy = np.mean(client_weighted_val_accuracies)
            accuracy_based_agg_weights.append(average_client_weighted_val_accuracy)

        return accuracy_based_agg_weights
    
    def serialize_state_dict(self, state_dict) -> bytes:
        buf = io.BytesIO()
        # save CPU tensors to avoid CUDA device issues during load
        cpu_state = {k: v.cpu() for k, v in state_dict.items()}
        torch.save(cpu_state, buf)
        return buf.getvalue()

    def deserialize_state_dict(self, b: bytes, map_location=None):
        buf = io.BytesIO(b)
        return torch.load(buf, map_location=map_location)
    
    def send_global_model_to_client(self, client_id, client_api_url, global_state_dict):
        # Serialize global model to bytes
        global_model_bytes = self.serialize_state_dict(global_state_dict)

        channel = grpc.insecure_channel(client_api_url)
        stub = ClientgRPC_pb2_grpc.ClientServiceStub(channel)

        request = ClientgRPC_pb2.ReceiveGlobalModelRequest(
            client_id=client_id,
            global_model=global_model_bytes
        )

        response = stub.ReceiveGlobalModel(request)

        # response contains client_model + training_time + iid_measure
        return response.client_model, response.training_time, response.iid_measure

    # ============================================================
    # Federated Learning Coordinator
    # ============================================================
    def start_federated_learning(self):
        # Start Timer
        total_training_start = time.time()
        rng = np.random.default_rng(42)

        # IID Based Aggregation Weights
        if self.fl_accuracy["selected_algorithm"] == "(1-JSD)":
            print("Calculating IID Aggregation Weights")
            for cid in self.fl_train_config["clients_ids"]:
                client_send_iid_measure_url = self.clients["client_api_urls"][cid] + "/send/iid-measure"
                iid_agg_weights = []
                try:
                    response = requests.get(
                        client_send_iid_measure_url,
                        json={"selected_algorithm": self.fl_accuracy["selected_algorithm"]},
                        timeout=None
                    )
                    response.raise_for_status()
                    iid_measure_result = response.json()
                    iid_agg_weight = iid_measure_result['iid_agg_weights']
                    iid_agg_weights_cal_time = iid_measure_result['iid_agg_weights_cal_time']
                    iid_agg_weights.append(iid_agg_weight)
                    self.history["iid_agg_weights_cal_time"].append(iid_agg_weights_cal_time)
                    print(f"✓ Received IID measure from Client {cid}")
                except Exception as e:
                    print(f"Failed to receive IID measure from Client {cid}: {e}")
            print("IID Aggregation Weights: ", iid_agg_weights)
        else:
            iid_agg_weights = None
            iid_agg_weights_cal_time = 0.0



        # Federated Learning Rounds
        for fl_round in range(self.fl_train_config["fl_rounds"]):
            print(f"\n{'='*80}\n FEDERATED LEARNING ROUND {fl_round + 1}/{self.fl_train_config["fl_rounds"]}\n{'='*80}")
            round_start_time = time.time()
            
            # Select clients
            selected_clients = rng.choice(self.clients["clients_ids"], size=self.fl_train_config["num_selected_clients"], replace=False)
            self.clients["selected_clients_ids"] = list(selected_clients)
            self.clients["selected_clients_models"] = {cid: [] for cid in selected_clients}

            # Serialize global model
            model_size_bytes = self.cal_size(self.global_model)
            self.history["global_model_size_bytes"].append(model_size_bytes)
            print(f"Global model size: {model_size_bytes / 1e6:.4f} MB")

            round_global_send_time = 0.0
            round_local_train_time = 0.0
            round_global_recv_time = 0.0

            print(f"Selected clients: {selected_clients}")

            for cid in selected_clients:
                print(f"{'='*80}\nClient {cid} Local Training\n{'='*80}")


                client_model_params, training_time, iid_agg_measure = self.send_global_model_to_client(cid, self.clients["client_api_urls"][cid], self.global_model.state_dict())
                self.clients["selected_clients_models"][cid].append(self.deserialize_model(client_model_params))
                round_local_train_time = round_local_train_time + training_time

                if iid_agg_weights is not None:
                    iid_agg_weights[cid] = iid_agg_measure
                print(f"✓ Received trained model from Client {cid}")
                
            # Record communication times
            self.history["round_global_model_send_time"].append(round_global_send_time)
            self.history["round_client_model_recv_time"].append(round_global_recv_time)
            self.history["round_local_training_time"].append(round_local_train_time)
            self.history["round_communication_time"].append(round_global_send_time + round_global_recv_time)
            

            # Accuracy Based Aggregation Weights
            acc_based_agg_weights = []
            if self.fl_accuracy["selected_algorithm"] == "AccuracyBased" or self.fl_accuracy["selected_algorithm"] == "AccuracyBased(1-JSD)" or self.fl_accuracy["selected_algorithm"] == "AccuracyBased_SEBW":
                print("Calculating Accuracy Based Aggregation Weights")
                for cid in selected_clients:
                    test_client_model_params = self.clients["selected_clients_models"][cid]
                    for t_cid in selected_clients:
                        if cid == t_cid:
                            continue
                        client_send_acc_based_measure_url = self.clients["client_api_urls"][cid] + "/send/acc-based-measure"
                        try:
                            response = requests.get(
                                client_send_acc_based_measure_url,
                                json={
                                    "client_model_params": test_client_model_params,
                                    "model_type": self.fl_train_config["model_type"],
                                },
                                timeout=None
                            )
                            response.raise_for_status()
                            acc_based_measure_result = response.json()
                            acc_based_agg_weights.append(acc_based_measure_result['acc_based_measure'])

                            print(f"✓ Received Accuracy Based measure from Client {cid}")
                        except Exception as e:
                            print(f"Failed to receive Accuracy Based measure from Client {cid}: {e}")
            else:
                acc_based_agg_weights = None
                        
            # Get Aggregation Weights
            if iid_agg_weights and acc_based_agg_weights:
                print("IID Measure Weights: ", iid_agg_weights)
                print("Accuracy Based Weight: ", acc_based_agg_weights)
                client_agg_weights = [
                    iid_agg_weights[cid] * acc_based_agg_weights[i] for i, cid in enumerate(selected_clients)
                ]
                print("Combined Weights: ", client_agg_weights)
            elif iid_agg_weights:
                print("IID Measure Weights: ", iid_agg_weights)
                client_agg_weights = [iid_agg_weights[cid] for cid in selected_clients]
            elif acc_based_agg_weights:
                print("Accuracy Based Weight: ", acc_based_agg_weights)
                client_agg_weights = [acc_based_agg_weights[i] for i in range(len(selected_clients))]
            else:
                client_agg_weights = None

            # Aggregate Models
            agg_start_time = time.time()
            self.aggregate_models(selected_clients, client_agg_weights)
            agg_time = time.time() - agg_start_time
            self.history["round_aggregation_time"].append(agg_time)

            # Evaluate Global Model
            acc, loss = self.test_model()
            self.history["global_accuracy"].append(acc)
            self.history["global_loss"].append(loss)
            round_time = time.time() - round_start_time
            self.history["fl_train_time"].append(round_time)
            system_usage = self.get_system_usage(self.device)
            self.history["system_usage"].append(system_usage)

            print(f"Round {fl_round+1}: Accuracy={acc:.2f}%, Loss={loss:.4f}")

            wandb.log({
                "fl_rounds": fl_round + 1,
                "global_accuracy": acc,
                "global_loss": loss,
                "round_communication_time": self.history["round_communication_time"][-1],
                "round_global_model_send_time": self.history["round_global_model_send_time"][-1],
                "round_client_model_recv_time": self.history["round_client_model_recv_time"][-1],
                "round_local_training_time": self.history["round_local_training_time"][-1],
                "round_aggregation_time": self.history["round_aggregation_time"][-1],
                "round_dequantization_time": 0,
                "round_total_time": self.history["fl_train_time"][-1],
                "cpu_percent": system_usage["cpu_percent"],
                "ram_percent": system_usage["ram_percent"],
                **({k: v for k, v in system_usage.items() if "gpu" in k}),
            })
                
        print(f"\nFederated Learning Completed in {(time.time()-total_training_start)/60:.2f} minutes!")

        # 🧾 Save results
        self.save_results(results=self.history, results_save_dir=self.fl_train_config["results_save_dir"])
        
        # Finish the WANDB
        wandb.finish()