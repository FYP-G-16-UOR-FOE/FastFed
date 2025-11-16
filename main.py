import multiprocessing
import os

import requests
import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from client.Client import Client
from dataset_partitioner.DatasetPartitioner import DatasetPartitioner
from models.NormalCNN import NormalCNN
from models.VGG import VGG
from server.Server import Server

# ============================================
# -------- GLOBAL CONFIGURATION --------------
# ============================================

# FL parameters
FL_ROUNDS = 150
LOCAL_EPOCHS = 4
NUM_CLIENTS = 20
NUM_SELECTED_CLIENTS = 10
SELECTED_MODEL = "VGG"  # "VGG", "NormalCNN"
SELECTED_PERFORMANCE_OPTIMIZATION_TYPES = [] # ["Quantization", "MixedPrecision", "GradientAccumulation"]
SELECTED_ALGORITHM = "FedAvg" # FedAvg, (1-JSD), ClientSize(1-JSD), AccuracyBased, AccuracyBased(1-JSD), SEBW, AccuracyBased_SEBW, FedProx, CAFA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DISTRIBUTION_TYPE = "Realworld_Distribution_C20_1"
RESULTS_SAVE_PATH = './history'
RESULTS_DIR = os.path.join(RESULTS_SAVE_PATH, "FL_Results")

# ============================================
# --------------- SERVER API -----------------
# ============================================

def create_server_api():
    app = FastAPI()

    # Initialize Global Model
    if SELECTED_MODEL == "NormalCNN":
        global_model = NormalCNN().to(DEVICE)
    else:
        global_model = VGG("VGG19").to(DEVICE)

    # Initialize Server
    server = Server(
        global_model=global_model,
        device=DEVICE,
        model_type=SELECTED_MODEL,
        fl_rounds=FL_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        num_clients=NUM_CLIENTS,
        num_selected_clients=NUM_SELECTED_CLIENTS,
        results_save_dir=RESULTS_DIR,
        selected_algorithm=SELECTED_ALGORITHM
    )

    class RegisterRequest(BaseModel):
        client_id: int
        client_api_url: str

    @app.post("/api/register")
    def register_client(req: RegisterRequest):
        if req.client_id in server.clients["clients_ids"]:
            return {"status": "already registered"}

        server.clients["clients_ids"].append(req.client_id)
        server.clients["client_api_urls"].append(req.client_api_url)
        return {"status": "registered"}

    @app.post("/api/start-federated-learning")
    async def start_federated_learning(background_tasks: BackgroundTasks):
        if len(server.clients["clients_ids"]) == NUM_CLIENTS:
            background_tasks.add_task(server.start_federated_learning)
        return {"status": "FL started in background"}

    return app

# ============================================
# --------------- CLIENT API -----------------
# ============================================

def create_client_api():
    app = FastAPI()

    # Initialize Global Model
    if SELECTED_MODEL == "NormalCNN":
        global_model = NormalCNN().to(DEVICE)
    else:
        global_model = VGG("VGG19").to(DEVICE)

    # Create client datasets
    dataset_partitioner = DatasetPartitioner(
        n_clients=NUM_CLIENTS,
        batch_size=64,
        max_class_per_client=10,
        seed=42,
        verbose=True,
    )
    client_datasets, _ = dataset_partitioner.split_cifar10_realworld()

    # Initialize Clients
    clients = [
        Client(
            id=i,
            ip_address="127.0.0.1",
            port=8000,
            device=DEVICE,
            model=global_model,
            client_dataset=client_datasets[i],
            batch_size=64
        ) for i in range(NUM_CLIENTS)
    ]

    class GlobalModelRequest(BaseModel):
        client_id: int
        global_model: dict
        model_type: str

    @app.post("/api/receive/global-model")
    def receive_global_model(req: GlobalModelRequest):
        client = clients[req.client_id]
        print("Receiving global model at Client:", req.client_id)
        client.set_global_model_to_client(req.global_model)
        print("Received global model")
        return {"status": "global model received"}

    class LocalTrainRequest(BaseModel):
        client_id: int
        epochs: int
        selected_algorithm: str
        is_use_full_dataset: bool
        model_type: str
        performance_optimization_types: list

    @app.post("/api/start/local-training")
    def start_local_training(req: LocalTrainRequest):
        print("Starting local training...")
        client = clients[req.client_id]
        client.train_client_model(
            epochs=req.epochs,
            selected_algorithm=req.selected_algorithm,
            is_use_full_dataset=req.is_use_full_dataset,
            model_type=req.model_type,
            performance_optimization_types=req.performance_optimization_types
        )
        return {"status": "local training started"}

    class GetTrainedModelRequest(BaseModel):
        client_id: int

    @app.get("/api/send/trained-model")
    def send_trained_model(req: GetTrainedModelRequest):
        print("Sending trained model")
        client = clients[req.client_id]
        trained_model = client.get_client_model_params()
        trained_duration = client.local_training_details["train_times"][-1] if client.local_training_details["train_times"] else None
        return {"trained_model": trained_model, "training_duration": trained_duration}

    # Register all clients with server
    print("Registering clients...")
    for client in clients:
        server_register_url = "http://127.0.0.1:9000/api/register"
        try:
            r = requests.post(server_register_url, json={
                "client_id": client.id,
                "client_api_url": "http://127.0.0.1:8000/api"
            })
            r.raise_for_status()
            print(f"Client {client.id} registered")
        except Exception as e:
            print(f"Failed to register client {client.id}: {e}")

    # AUTO START FL
    try:
        requests.post("http://127.0.0.1:9000/api/start-federated-learning")
    except:
        pass

    return app

# ============================================
# ----------- START BOTH SERVERS ------------
# ============================================

def run_server_api():
    uvicorn.run(create_server_api(), host="127.0.0.1", port=9000)

def run_client_api():
    uvicorn.run(create_client_api(), host="127.0.0.1", port=8000)

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=run_server_api)
    p2 = multiprocessing.Process(target=run_client_api)

    p1.start()
    p2.start()

    try:
        p1.join()
        p2.join()
    except KeyboardInterrupt:
        print("Stopping servers...")
        p1.terminate()
        p2.terminate()
        p1.join()
        p2.join()
        print("Servers stopped.")