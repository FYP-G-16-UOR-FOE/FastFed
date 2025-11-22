"""
Corrected federated learning gRPC server+client runner
- Moves side-effect code out of class bodies
- Runs server FL loop after gRPC server is started (in background thread)
- Serializes model state_dict to bytes for sending over gRPC
- Uses deepcopy for per-client model copies
- Uses multiprocessing spawn start method to avoid CUDA reinit issues
- Adds retries/wait for server when clients register

Assumes the generated protobuf modules exist and that message fields for model bytes
are declared as `bytes` in the proto (e.g. `bytes client_model = 3;`).
"""

import copy
import io
import multiprocessing
import os
import sys
import threading
import time
from concurrent import futures
from typing import List

import grpc
import torch

import wandb
# Local imports (assumed present)
from client.Client import Client
from dataset_partitioner.DatasetPartitioner import DatasetPartitioner
from gRPC import (ClientgRPC_pb2, ClientgRPC_pb2_grpc, ServergRPC_pb2,
                  ServergRPC_pb2_grpc)
from models.NormalCNN import NormalCNN
from models.VGG import VGG
from server.Server import Server

# ----------------- Configuration -----------------
# gRPC max message sizes (bytes) - increase because model state_dicts can be large
MAX_MESSAGE_LENGTH = 500 * 1024 * 1024  # 500 MB
FL_ROUNDS = 150
LOCAL_EPOCHS = 4
NUM_CLIENTS = 20
NUM_SELECTED_CLIENTS = 10
DATASET_PARTITIONER_MAX_CLASS_PER_CLIENT = 10
DATASET_PARTITIONER_SEED = 42
IS_USE_FULL_DATASET = False
SELECTED_MODEL = "VGG"  # "VGG" or "NormalCNN"
SELECTED_ALGORITHM = "FedAvg"  # FedAvg, (1-JSD), ClientSize(1-JSD), AccuracyBased, AccuracyBased(1-JSD), SEBW, AccuracyBased_SEBW, FedProx, CAFA
FEDPROX_MU = 0.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_SAVE_PATH = '/content/drive/MyDrive/Final_Implementations/FL_Experiments'
RESULTS_DIR = os.path.join(RESULTS_SAVE_PATH, "FL_Results")
BATCH_SIZE = 64
SERVER_HOST = 'localhost'
SERVER_PORT = 50051
CLIENT_BASE_PORT = 60000

# ----------------- Helpers -----------------

def serialize_state_dict(state_dict) -> bytes:
    buf = io.BytesIO()
    # save CPU tensors to avoid CUDA device issues during load
    cpu_state = {k: v.cpu() for k, v in state_dict.items()}
    torch.save(cpu_state, buf)
    return buf.getvalue()


def deserialize_state_dict(b: bytes, map_location=None):
    buf = io.BytesIO(b)
    return torch.load(buf, map_location=map_location)

# ----------------- Server Servicer -----------------
class ServerServicer(ServergRPC_pb2_grpc.ServerServiceServicer):
    def __init__(self):
        # initialize server object used for FL logic
        if SELECTED_MODEL == "NormalCNN":
            global_model = NormalCNN().to(DEVICE)
        else:
            global_model = VGG("VGG19").to(DEVICE)

        self.server = Server(
            global_model=global_model,
            device=DEVICE,
            model_type=SELECTED_MODEL,
            fl_rounds=FL_ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            num_clients=NUM_CLIENTS,
            num_selected_clients=NUM_SELECTED_CLIENTS,
            results_save_dir=RESULTS_DIR,
            selected_algorithm=SELECTED_ALGORITHM,
        )

    def RegisterClient(self, request, context):
        client_id = request.client_id
        client_api_url = request.client_api_url
        print(f"[Server] RegisterClient: id={client_id} url={client_api_url}")
        try:
            self.server.register_client(client_id, client_api_url)
            return ServergRPC_pb2.StatusResponse(status="OK")
        except Exception as e:
            print("[Server] Error registering client:", e, file=sys.stderr)
            return ServergRPC_pb2.StatusResponse(status=f"ERROR: {e}")

# ----------------- Client Servicer -----------------
class ClientServicer(ClientgRPC_pb2_grpc.ClientServiceServicer):
    def __init__(self, clients: List[Client]):
        self.clients = clients

    def ReceiveGlobalModel(self, request, context):
        client_id = request.client_id
        model_bytes = request.global_model
        print(f"[ClientServicer] ReceiveGlobalModel for client {client_id}")
        try:
            state_dict = deserialize_state_dict(model_bytes, map_location=self.clients[client_id].device)
            self.clients[client_id].receive_global_model(state_dict)
            return ClientgRPC_pb2.StatusResponse(status="OK")
        except Exception as e:
            print("[Client] Error applying global model:", e, file=sys.stderr)
            raise e
        
    def StartLocalTraining(self, request, context):
        client_id = request.client_id
        print(f"[ClientServicer] StartLocalTraining for client {client_id}")
        try:
            self.clients[client_id].start_client_local_training()
            return ClientgRPC_pb2.StatusResponse(status="OK")
        except Exception as e:
            print("[Client] Error during local training:", e, file=sys.stderr)
            raise e
        
    def GetClientsTrainedModel(self, request, context):
        client_id = request.client_id
        print(f"[ClientServicer] GetClientsTrainedModel for client {client_id}")
        try:
            client_model_params, training_time = self.clients[client_id].get_client_updates()
            trained_model_bytes = serialize_state_dict(client_model_params)
            return ClientgRPC_pb2.GetClientsTrainedModelResponse(
                client_id=client_id,
                trained_model=trained_model_bytes,
                training_time=training_time
            )
        except Exception as e:
            print("[Client] Error getting trained model:", e, file=sys.stderr)
            raise e
        
    def GetIIDMeasure(self, request, context):
        client_id = request.client_id
        print(f"[ClientServicer] GetIIDMeasure for client {client_id}")
        try:
            iid_measure = self.clients[client_id].calculate_iid_measure()
            return ClientgRPC_pb2.GetIIDMeasureResponse(
                client_id=client_id,
                iid_measure=iid_measure
            )
        except Exception as e:
            print("[Client] Error calculating IID measure:", e, file=sys.stderr)
            raise e

    def ReceiveModelForAccuracyBasedMeasure(self, request, context):
        client_id = request.client_id
        model_bytes = request.model
        try:
            state_dict = deserialize_state_dict(model_bytes, map_location=self.clients[client_id].device)
            weighted_val_acc = self.clients[client_id].evaluate_state_dict_on_validation_data(state_dict)
            return ClientgRPC_pb2.AccuracyBasedMeasureResponse(weighted_val_acc=weighted_val_acc)
        except Exception as e:
            print("[Client] Error computing accuracy-based measure:", e, file=sys.stderr)
            raise e

# ----------------- Server runner -----------------

def server_serve():
    # create gRPC server with increased message size limits
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]
    )
    servicer = ServerServicer()
    ServergRPC_pb2_grpc.add_ServerServiceServicer_to_server(servicer, grpc_server)
    server_address = f"[::]:{SERVER_PORT}"
    grpc_server.add_insecure_port(server_address)
    grpc_server.start()
    print(f"[Server] gRPC server started on {server_address}")

    def wait_until_clients_register(server, n=NUM_CLIENTS):
        while len(server.clients["clients_ids"]) < n:
            print(f"[Server] Waiting for clients... {len(server.clients['clients_ids'])}/{n}")
            time.sleep(1)

    # Run federated learning in a background thread so gRPC can still accept RPCs
    # Wait for client registrations
    wait_until_clients_register(servicer.server)

    # 🧭 Initialize wandb
    wandb.login(key="cca506f824e9db910b7b4a407afc0b36ba655c28")
    wandb.init(
        project=f"FL-Test_gRPC_1",
        config={
            "model": SELECTED_MODEL,
            "fl_rounds": FL_ROUNDS,
            "clients": NUM_CLIENTS,
            "selected_clients": NUM_SELECTED_CLIENTS,
            "local_epochs": LOCAL_EPOCHS,
            "dataset_distribution": "Realistic Non-IID",
            "device": str(DEVICE)
        },
        name=SELECTED_ALGORITHM
    )   

    # Start FL loop
    fl_thread = threading.Thread(target=servicer.server.start_federated_learning, daemon=True)
    fl_thread.start()
    print("[Server] Federated learning loop started in background thread")

    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        print("[Server] Shutting down...")
        grpc_server.stop(0)

# ----------------- Client runner -----------------

def client_serve(client_start_port=CLIENT_BASE_PORT):
    # Prepare datasets and client objects (no side-effects in class bodies)
    dataset_partitioner = DatasetPartitioner(
        n_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
        max_class_per_client=DATASET_PARTITIONER_MAX_CLASS_PER_CLIENT,
        seed=DATASET_PARTITIONER_SEED,
        verbose=True,
    )
    client_datasets, _ = dataset_partitioner.split_cifar10_realworld()

    # create deep copies of the base model for each client
    if SELECTED_MODEL == "NormalCNN":
        base_model = NormalCNN()
    else:
        base_model = VGG("VGG19")

    clients = []
    for i in range(NUM_CLIENTS):
        client_model = copy.deepcopy(base_model)
        c = Client(
            id=i,
            ip_address='127.0.0.1',
            port=client_start_port,
            device=DEVICE,
            model=client_model,
            client_dataset=client_datasets[i],
            batch_size=BATCH_SIZE,
            selected_algorithm=SELECTED_ALGORITHM,
            is_use_full_dataset=IS_USE_FULL_DATASET,
            model_type=SELECTED_MODEL,
            local_epochs=LOCAL_EPOCHS,
            fedprox_mu=FEDPROX_MU,
        )
        clients.append(c)

    # Start client-side gRPC server so server can call back (optional depending on architecture)
    # create gRPC server with increased message size limits
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]
    )
    servicer = ClientServicer(clients)
    ClientgRPC_pb2_grpc.add_ClientServiceServicer_to_server(servicer, grpc_server)
    client_server_port = client_start_port
    client_address = f"[::]:{client_server_port}"
    grpc_server.add_insecure_port(client_address)
    grpc_server.start()
    print(f"[Client] Client gRPC server started on {client_address}")

    # Register clients with the central server (retry until server available)
    # create channel to server with increased message size limits
    server_channel = grpc.insecure_channel(
        f"{SERVER_HOST}:{SERVER_PORT}",
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]
    )
    stub = ServergRPC_pb2_grpc.ServerServiceStub(server_channel)

    for client in clients:
        attempt = 0
        while True:
            try:
                # provide a callback URL or port so the server can reach back if needed
                client_api_url = f"{client.ip_address}:{client.port}"
                req = ServergRPC_pb2.ClientRegistrationRequest(client_id=client.id, client_api_url=client_api_url)
                resp = stub.RegisterClient(req)
                print(f"[Client] Registered client {client.id} -> {resp.status}")
                break
            except Exception as e:
                attempt += 1
                if attempt > 20:
                    print(f"[Client] Failed to register client {client.id} after {attempt} attempts: {e}", file=sys.stderr)
                    break
                print(f"[Client] Waiting for server to be ready... (attempt {attempt})")
                time.sleep(1.0)

    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        print("[Client] Shutting down...")
        grpc_server.stop(0)

# ----------------- Entrypoint -----------------

def serve():
    # Use spawn on platforms where CUDA or torch interactions are sensitive
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # already set
        pass

    p_server = multiprocessing.Process(target=server_serve)
    p_client = multiprocessing.Process(target=client_serve)

    p_server.start()
    # small sleep to increase chance server is listening before clients attempt registration
    time.sleep(1.0)
    p_client.start()

    try:
        p_server.join()
        p_client.join()
    except KeyboardInterrupt:
        print("[Main] Terminating child processes...")
        p_server.terminate()
        p_client.terminate()


if __name__ == '__main__':
    serve()
