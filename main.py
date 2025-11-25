import copy
import multiprocessing
import sys
import threading
import time
from concurrent import futures

import grpc
import torch

import wandb
from client.Client import Client
from client.ClientServicer import ClientServicer
from dataset_partitioner.DatasetPartitioner import DatasetPartitioner
from gRPC import (ClientgRPC_pb2, ClientgRPC_pb2_grpc, ServergRPC_pb2,
                  ServergRPC_pb2_grpc)
from models.NormalCNN import NormalCNN
from models.VGG import VGG
from server.ServerServicer import ServerServicer
from utils.ParseArgument import ParseArgument

# ----------------- Server runner -----------------

def server_serve(config):
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10)
    )
    servicer = ServerServicer(config)
    ServergRPC_pb2_grpc.add_ServerServiceServicer_to_server(servicer, grpc_server)
    server_address = f"{config['server_host']}:{config['server_port']}"
    grpc_server.add_insecure_port(server_address)
    grpc_server.start()
    print(f"[Server] gRPC server started on {server_address}")

    def wait_until_clients_register(server, n):
        while len(server.clients["clients_ids"]) < n:
            print(f"[Server] Waiting for clients... {len(server.clients['clients_ids'])}/{n}")
            time.sleep(2.0)

    # Wait for client registrations
    wait_until_clients_register(servicer.server, config["num_clients"])

    # 🧭 Initialize wandb
    if config["wandb_key"]:
        wandb.login(key=config["wandb_key"])
    wandb.init(
        project=config["wandb_project"],
        config={
            "model": config["model"],
            "fl_rounds": config["fl_rounds"],
            "clients": config["num_clients"],
            "selected_clients": config["num_selected_clients"],
            "local_epochs": config["local_epochs"],
            "dataset_distribution": "Realistic Non-IID",
            "device": str(config["device"]),
            "algorithm": config["algorithm"],
        },
        name=config["algorithm"]
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

def client_serve(config):
    
    # Prepare datasets and client objects (no side-effects in class bodies)
    dataset_partitioner = DatasetPartitioner(
        n_clients=config["num_clients"],
        batch_size=config["batch_size"],
        max_class_per_client=config["max_class_per_client"],
        seed=config["seed"],
        verbose=True,
    )
    client_datasets, _ = dataset_partitioner.split_cifar10_realworld()

    # create deep copies of the base model for each client
    if config["model"] == "NormalCNN":
        base_model = NormalCNN()
    else:
        base_model = VGG("VGG19")

    clients = []
    for i in range(config["num_clients"]):
        client_model = copy.deepcopy(base_model)
        c = Client(
            id=i,
            ip_address=config["server_host"],
            port=config["client_base_port"] + i,
            device=config["device"],
            model=client_model,
            client_dataset=client_datasets[i],
            batch_size=config["batch_size"],
            selected_algorithm=config["algorithm"],
            is_use_full_dataset=config["use_full_dataset"],
            model_type=config["model"],
            local_epochs=config["local_epochs"],
            fedprox_mu=config["fedprox_mu"],
        )
        clients.append(c)

    # Start gRPC servers for each client
    grpc_servers = []

    for client in clients:
        grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
        )
        servicer = ClientServicer(client)
        ClientgRPC_pb2_grpc.add_ClientServiceServicer_to_server(servicer, grpc_server)

        client_address = f"{client.ip_address}:{client.port}"
        grpc_server.add_insecure_port(client_address)
        grpc_server.start()
        grpc_servers.append(grpc_server)

        print(f"[Client-{client.id}] gRPC server started on {client_address}")

    # Register clients with the central server
    server_channel = grpc.insecure_channel(
        f"{config['server_host']}:{config['server_port']}",
    )
    stub = ServergRPC_pb2_grpc.ServerServiceStub(server_channel)

    for client in clients:
        attempt = 0
        while True:
            try:
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
                time.sleep(2)

    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        print("[Client] Shutting down...")
        grpc_server.stop(0)

# ----------------- Entrypoint -----------------

def serve(config):
    # Use spawn on platforms where CUDA or torch interactions are sensitive
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # already set
        pass

    p_server = multiprocessing.Process(target=server_serve, args=(config,))
    p_client = multiprocessing.Process(target=client_serve, args=(config,))

    p_server.start()
    # small sleep to increase chance server is listening before clients attempt registration
    time.sleep(2.0)
    p_client.start()

    try:
        p_server.join()
        p_client.join()
    except KeyboardInterrupt:
        print("[Main] Terminating child processes...")
        p_server.terminate()
        p_client.terminate()


if __name__ == '__main__':
    args = ParseArgument.parse_arguments()
    config = {
        "model": args.model,
        "algorithm": args.algorithm,
        "fl_rounds": args.fl_rounds,
        "local_epochs": args.local_epochs,
        "num_clients": args.num_clients,
        "num_selected_clients": args.num_selected_clients,
        "batch_size": args.batch_size,
        "fedprox_mu": args.fedprox_mu,
        "max_class_per_client": args.max_class_per_client,
        "use_full_dataset": args.use_full_dataset,
        "device": torch.device(args.device),
        "server_host": args.server_host,
        "server_port": args.server_port,
        "client_base_port": args.client_base_port,
        "results_dir": args.results_dir,
        "wandb_project": args.wandb_project,
        "wandb_key": args.wandb_key,
        "seed": args.seed,
    }
    serve(config)
