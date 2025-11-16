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

FL_ROUNDS = 150
LOCAL_EPOCHS = 1
NUM_CLIENTS = 20
NUM_SELECTED_CLIENTS = 2
SELECTED_MODEL = "VGG"
SELECTED_ALGORITHM = "FedAvg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_SAVE_PATH = './history'
RESULTS_DIR = os.path.join(RESULTS_SAVE_PATH, "FL_Results")

# ============================================
# --------------- FASTAPI APP ----------------
# ============================================

app = FastAPI()

# ----------------- Server Setup -----------------
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

# ----------------- Client Setup -----------------
# Initialize client datasets
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
        port=9000,
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
    print(f"Receiving global model at Client: {req.client_id}")
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
    client = clients[req.client_id]
    print(f"Starting local training for Client {req.client_id}...")
    duration = client.train_client_model(
        epochs=req.epochs,
        selected_algorithm=req.selected_algorithm,
        is_use_full_dataset=req.is_use_full_dataset,
        model_type=req.model_type,
        performance_optimization_types=req.performance_optimization_types
    )
    return {"status": "local training started", "training_duration": duration}

class GetTrainedModelRequest(BaseModel):
    client_id: int

@app.get("/api/send/trained-model")
def send_trained_model(req: GetTrainedModelRequest):
    client = clients[req.client_id]
    trained_model = client.get_client_model_params()
    trained_duration = client.local_training_details["train_times"][-1] if client.local_training_details.get("train_times") else None
    return {"trained_model": trained_model, "training_duration": trained_duration}

# ----------------- Auto Register Clients -----------------
print("Registering clients to server...")
for client in clients:
    """ server_register_url = "http://127.0.0.1:9000/api/register"
    try:
        r = requests.post(server_register_url, json={
            "client_id": client.id,
            "client_api_url": "http://127.0.0.1:9000/api"
        })
        r.raise_for_status()
        print(f"Client {client.id} registered")
    except Exception as e:
        print(f"Failed to register client {client.id}: {e}")
 """
    server.clients["clients_ids"].append(client.id)
    server.clients["client_api_urls"].append(f"http://127.0.0.1:9000/api")


# Auto start FL
try:
    requests.post("http://127.0.0.1:9000/api/start-federated-learning")
except:
    pass

# ============================================
# ----------------- RUN APP ------------------
# ============================================

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
    