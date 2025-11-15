import os

import NormalCNN
import requests
import torch
import VGG
from fastapi import FastAPI
from pydantic import BaseModel

from client.Client import Client
from dataset_partitioner.DatasetPartitioner import DatasetPartitioner

app = FastAPI()


SELECTED_MODEL = "VGG" # "VGG", "NormalCNN"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 20

# Initialize the Global model and Server
if SELECTED_MODEL == "NormalCNN":
    global_model = NormalCNN().to(DEVICE)
else:
    global_model = VGG('VGG19').to(DEVICE)

# Create clients datasets
dataset_partitioner = DatasetPartitioner(
    num_clients=NUM_CLIENTS,
    batch_size=64,
    max_class_per_client=20,
    seed=42,
    verbose=True,
)
client_datasets, _ = dataset_partitioner.split_cifar10_realworld()

# Initialize Clients
clients = [
    Client(
        id=i,
        ip_address="localhost",
        port=8000 + i, 
        device=DEVICE, 
        model=global_model,
        client_dataset=client_datasets[i],
        batch_size=64
    )
    for i in range(NUM_CLIENTS)]


class GlobalModelRequest(BaseModel):
    client_id: int
    global_model: dict
    model_type: str

@app.post("api/receive/global-model")
def receive_global_model(req: GlobalModelRequest):
    client = clients[req.client_id]
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

@app.post("api/start/local-training")
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

@app.get("api/send/trained-model")
def send_trained_model(req):
    print("Sending trained model")
    client = clients[req.client_id]
    trained_model = client.get_client_model_params()
    return {"trained_model": trained_model}

# Register Clients
for client in clients:
    print(f"Registering Client ID: {client.id} at {client.ip_address}:{client.port}")
    server_register_url = "http:/localhost:9000/register"
    
    try:
        response = requests.post(
            url=server_register_url,
            json={"client_id": client.id, "client_api_url": f"http://{client.ip_address}:{client.port}/api"})
        response.raise_for_status()
        print(f"Client ID: {client.id} registered successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to register Client ID: {client.id}. Error: {e}")