import collections
import copy
import io
import os
import pickle
import time
from collections import Counter

import grpc
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
from gRPC import ClientgRPC_pb2, ClientgRPC_pb2_grpc

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
            "total_iid_agg_weights_cal_time": 0.0,
            "acc_based_agg_weights_cal_time": [],
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
        Perform standard FedAvg aggregation on a list of parameter OrderedDicts.
        Handles NumPy arrays, PyTorch tensors, and scalars.
        """
        avg_params = collections.OrderedDict()

        for key in weights_list[0].keys():
            vals = []

            for w in weights_list:
                v = w[key]

                # Convert PyTorch tensor → numpy
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()

                # Convert scalar → ndarray
                if np.isscalar(v):
                    v = np.array(v)

                vals.append(v)

            stacked = np.stack(vals)
            avg_params[key] = torch.tensor(np.mean(stacked, axis=0))

        return avg_params
    
    def weighted_fed_avg(self, weights_list, agg_weights):
        """
        Weighted FedAvg (handles arrays, tensors, scalars).
        """
        total_weight = sum(agg_weights)
        avg_params = collections.OrderedDict()

        for key in weights_list[0].keys():
            vals = []

            for i, w in enumerate(weights_list):
                v = w[key]

                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()

                if np.isscalar(v):
                    v = np.array(v)

                vals.append(v * agg_weights[i])

            stacked = np.stack(vals)
            avg_params[key] = torch.tensor(np.sum(stacked, axis=0) / total_weight)

        return avg_params
    
    def aggregate_models(self, client_models=None, agg_weights=None):
        if client_models is None:
            client_models = self.clients["selected_clients_models"]

        weights_list = []

        for client_model in client_models:
            state_dict = client_model.state_dict() if hasattr(client_model, "state_dict") else client_model
            weights_np = collections.OrderedDict()

            for k, v in state_dict.items():

                # Tensor → numpy or scalar numpy
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()

                # int/float → numpy array
                if np.isscalar(v):
                    v = np.array(v)

                weights_np[k] = v

            weights_list.append(weights_np)

        # ----------------------------------------------------------
        # Select aggregation method
        # ----------------------------------------------------------
        if not agg_weights or (
            len(agg_weights) > 0 and all(abs(agg_weights[i] - agg_weights[0]) < 1e-6
                                        for i in range(len(agg_weights)))
        ):
            print("\n🟢 Using FedAvg aggregation.")
            aggregated_model_params = self.fed_avg(weights_list)
        else:
            print("\n🟡 Using Weighted FedAvg aggregation.")
            print("Aggregation Weights:", agg_weights)
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
            raise ValueError(f"Invalid model type: {self.fl_train_config["model_type"]}.")

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
                    "model_type": self.fl_train_config["model_type"],
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
        # Serialize model
        global_model_bytes = self.serialize_state_dict(global_state_dict)
        print(f"Serialized global model size (MB): {len(global_model_bytes)/1e6:.2f}")

        # Prepare the GRPC client
        channel = grpc.insecure_channel(client_api_url)
        stub = ClientgRPC_pb2_grpc.ClientServiceStub(channel)

        # ---- Generator for streaming data ----
        def request_generator():
            # First send metadata (client ID)
            yield ClientgRPC_pb2.ReceiveGlobalModelRequest(
                client_id=client_id
            )

            # Send the model in chunks
            chunk_size = 1*1024*1024  # 1 MB
            for i in range(0, len(global_model_bytes), chunk_size):
                chunk = global_model_bytes[i:i+chunk_size]
                yield ClientgRPC_pb2.ReceiveGlobalModelRequest(global_model=chunk)

        # ---- Send the stream ----
        response = stub.ReceiveGlobalModel(request_generator())

        print(f"✓ Sent global model to Client {client_id}")
        return response

    
    def start_client_local_training(self, client_id, client_api_url):
        channel = grpc.insecure_channel(client_api_url)
        stub = ClientgRPC_pb2_grpc.ClientServiceStub(channel)

        request = ClientgRPC_pb2.StartLocalTrainingRequest(
            client_id=client_id
        )
        response = stub.StartLocalTraining(request)
        print(f"✓ Started local training for Client {client_id}")

    def get_client_updates(self, client_id, client_api_url):
        channel = grpc.insecure_channel(client_api_url)
        stub = ClientgRPC_pb2_grpc.ClientServiceStub(channel)

        # Send request
        request = ClientgRPC_pb2.GetClientsTrainedModelRequest(
            client_id=client_id
        )

        # Initialize
        client_model_bytes = b""
        training_time = None

        # Streamed response
        for i, message in enumerate(stub.GetClientsTrainedModel(request)):
            if i == 0:
                # First packet contains training_time
                training_time = message.training_time
            
            # Append model bytes
            client_model_bytes += message.trained_model

        # Now decode the final assembled model
        client_state_dict = self.deserialize_state_dict(
            client_model_bytes,
            map_location=self.device
        )
        print(f"✓ Received trained model from Client {client_id}")
        return client_state_dict, training_time
    
    def get_client_iid_measure(self, client_id, client_api_url):
        channel = grpc.insecure_channel(client_api_url)
        stub = ClientgRPC_pb2_grpc.ClientServiceStub(channel)

        request = ClientgRPC_pb2.GetIIDMeasureRequest(
            client_id=client_id
        )
        response = stub.GetIIDMeasure(request)
        return response.iid_measure
    
    def send_client_model_to_cal_val_acc(self, client_id, test_client_id, client_api_url, client_state_dict):
        # Serialize model
        client_model_bytes = self.serialize_state_dict(client_state_dict)
        print(f"Serialized global model size (MB): {len(client_model_bytes)/1e6:.2f}")

        # Prepare the GRPC client
        channel = grpc.insecure_channel(client_api_url)
        stub = ClientgRPC_pb2_grpc.ClientServiceStub(channel)

        # ---- Generator for streaming data ----
        def request_generator():
            # First send metadata (client ID)
            yield ClientgRPC_pb2.AccuracyBasedMeasureRequest(
                client_id=test_client_id,
                model_type=self.fl_train_config["model_type"]
            )

            # Send the model in chunks
            chunk_size = 1*1024*1024  # 1 MB
            for i in range(0, len(client_model_bytes), chunk_size):
                chunk = client_model_bytes[i:i+chunk_size]
                yield ClientgRPC_pb2.AccuracyBasedMeasureRequest(model=chunk)

        # ---- Send the stream (correct RPC method) ----
        response = stub.ReceiveModelForAccuracyBasedMeasure(request_generator())

        weighted_val_acc = response.weighted_val_acc
        print(f"Client {client_id} received weighted val accuracy from Client {test_client_id}: {weighted_val_acc:.4f}")
        return weighted_val_acc

    # ============================================================
    # Federated Learning Coordinator
    # ============================================================
    def start_federated_learning(self):
        # Start Timer
        total_training_start = time.time()
        rng = np.random.default_rng(42)

        # Get clients iid measure
        if self.fl_accuracy["selected_algorithm"] == "(1-JSD)" or self.fl_accuracy["selected_algorithm"] == "AccuracyBased(1-JSD)" or self.fl_accuracy["selected_algorithm"] == "SEBW" or self.fl_accuracy["selected_algorithm"] == "AccuracyBased_SEBW":
            print("\nGetting IID Measures from selected clients...")
            iid_agg_weights_cal_time = time.time()
            iid_agg_weights = []
            for cid in self.clients["clients_ids"]:
                iid_measure = self.get_client_iid_measure(cid, self.clients["client_api_urls"][cid])
                iid_agg_weights.append(iid_measure)
            iid_agg_weights_cal_time = time.time() - iid_agg_weights_cal_time
            self.history["total_iid_agg_weights_cal_time"] = iid_agg_weights_cal_time
            print("IID Aggregation Weights: ", iid_agg_weights)
        else:
            iid_agg_weights = None

        # Federated Learning Rounds
        for fl_round in range(self.fl_train_config["fl_rounds"]):
            print(f"\n{'='*80}\n FEDERATED LEARNING ROUND {fl_round + 1}/{self.fl_train_config["fl_rounds"]}\n{'='*80}")
            round_start_time = time.time()
            
            # Select clients
            selected_clients = rng.choice(self.clients["clients_ids"], size=self.fl_train_config["num_selected_clients"], replace=False)
            self.clients["selected_clients_ids"] = list(selected_clients)
            self.clients["selected_clients_models"] = {cid: [] for cid in selected_clients}

            # Reset round communication times
            self.clients["selected_clients_models"] = []
            round_global_send_time = 0.0
            round_local_train_time = 0.0
            round_client_model_recv_time = 0.0
            round_client_training_start_time = 0.0

            print(f"Selected clients: {selected_clients}")

            for cid in selected_clients:
                print(f"\n{'='*70}\nClient {cid} Local Training\n{'='*70}")

                # Send global model to client
                send_start_time = time.time()
                self.send_global_model_to_client(cid, self.clients["client_api_urls"][cid], self.global_model.state_dict())
                send_time = time.time() - send_start_time
                round_global_send_time += send_time

                # Start local training on client
                recv_start_time = time.time()
                self.start_client_local_training(cid, self.clients["client_api_urls"][cid])
                recv_time = time.time() - recv_start_time
                round_client_training_start_time += recv_time
                
                # Get client updates
                recv_time = time.time()
                client_model_params, training_time = self.get_client_updates(cid, self.clients["client_api_urls"][cid])
                round_client_model_recv_time += time.time() - recv_time
                self.clients["selected_clients_models"].append(client_model_params)
                round_local_train_time += training_time
                
            # Record communication times
            self.history["round_global_model_send_time"].append(round_global_send_time)
            self.history["round_client_model_recv_time"].append(round_client_model_recv_time)
            self.history["round_local_training_time"].append(round_local_train_time)
            self.history["round_communication_time"].append(round_global_send_time + round_client_model_recv_time)
            

            # Accuracy Based Aggregation Weights
            acc_based_agg_weights = []
            if self.fl_accuracy["selected_algorithm"] == "AccuracyBased" or self.fl_accuracy["selected_algorithm"] == "AccuracyBased(1-JSD)" or self.fl_accuracy["selected_algorithm"] == "AccuracyBased_SEBW" or self.fl_accuracy["selected_algorithm"] == "CAFA":
                print("Calculating Accuracy Based Aggregation Weights")
                acc_based_agg_weights_cal_time = time.time()
                for cid in selected_clients:
                    client_state_dict = self.clients["selected_clients_models"][selected_clients.index(cid)]
                    weighted_val_acc_list = []
                    for t_cid in selected_clients:
                        if cid == t_cid:
                            continue
                        try:
                            weighted_val_acc = self.send_client_model_to_cal_val_acc(
                                client_id = cid,
                                test_client_id=t_cid,
                                client_api_url=self.clients["client_api_urls"][t_cid],
                                client_state_dict=client_state_dict
                            )
                            weighted_val_acc_list.append(weighted_val_acc)
                        except Exception as e:
                            raise e
                    acc_based_agg_weights.append(np.mean(weighted_val_acc_list))

                acc_based_agg_weights_cal_time = time.time() - acc_based_agg_weights_cal_time
            else:
                acc_based_agg_weights = None
                acc_based_agg_weights_cal_time = 0.0
            self.history["acc_based_agg_weights_cal_time"].append(acc_based_agg_weights_cal_time)
            self.history["round_communication_time"][-1] += acc_based_agg_weights_cal_time
                        
            # Get Aggregation Weights
            if iid_agg_weights and acc_based_agg_weights:
                print("IID Measure Weights: ", iid_agg_weights)
                print("Accuracy Based Weight: ", acc_based_agg_weights)
                client_agg_weights = [
                    iid_agg_weights[cid] * acc_based_agg_weights[i] for i, cid in enumerate(selected_clients)
                ]
                print("Combined Weights: ", client_agg_weights)
            elif iid_agg_weights:
                client_agg_weights = [iid_agg_weights[cid] for cid in selected_clients]
                print("IID Measure Weights: ", client_agg_weights)
            elif acc_based_agg_weights:
                print("Accuracy Based Weight: ", acc_based_agg_weights)
                client_agg_weights = [acc_based_agg_weights[i] for i in range(len(selected_clients))]
            else:
                client_agg_weights = None

            # Aggregate Models
            agg_start_time = time.time()
            self.aggregate_models(self.clients["selected_clients_models"], client_agg_weights)
            agg_time = time.time() - agg_start_time
            self.history["round_aggregation_time"].append(agg_time)
            print(f"✓ Aggregated client models in {agg_time:.2f} seconds.")

            # Evaluate Global Model
            print("Evaluating global model...")
            acc, loss = self.test_model()
            self.history["global_accuracy"].append(acc)
            self.history["global_loss"].append(loss)
            round_time = time.time() - round_start_time
            self.history["fl_train_time"].append(round_time)
            system_usage = self.get_system_usage(self.device)
            self.history["system_usage"].append(system_usage)

            print(f"\n=============== Round {fl_round + 1} Summary ===============")
            print(f"Round {fl_round+1}: Global Model: Accuracy={acc:.2f}%, Loss={loss:.4f}")
            print(f"Round {fl_round+1}: FL Round Total Time={round_time:.2f} seconds")
            print(f"Round {fl_round+1}: Total Clients Training Time={round_local_train_time:.2f} seconds")
            print(f"Round {fl_round+1}: Round Communication Time={self.history['round_communication_time'][-1]:.2f} seconds")
            print("===============================================================")

            # Log to WANDB
            print(f"Logging to wandb...")
            wandb.log({
                "fl_rounds": fl_round + 1,
                "global_accuracy": acc,
                "global_loss": loss,
                "round_communication_time": self.history["round_communication_time"][-1],
                "round_global_model_send_time": self.history["round_global_model_send_time"][-1],
                "round_client_model_recv_time": self.history["round_client_model_recv_time"][-1],
                "round_local_training_time": self.history["round_local_training_time"][-1],
                "round_aggregation_time": self.history["round_aggregation_time"][-1],
                "iid_agg_weights_calc_time": self.history["iid_agg_weights_calc_time"],
                "acc_based_agg_weights_cal_time": self.history["acc_based_agg_weights_cal_time"][-1],
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