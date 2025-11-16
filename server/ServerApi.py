import os

import torch
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

import wandb
from models.NormalCNN import NormalCNN
from models.VGG import VGG
from server.Server import Server

app = FastAPI()

# Parameters
FL_ROUNDS = 150
LOCAL_EPOCHS = 4
NUM_CLIENTS = 20
NUM_SELECTED_CLIENTS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DISTRIBUTION_TYPE = "Realworld_Distribution_C20_1"
RESULTS_SAVE_PATH = '/content/drive/MyDrive/history'
SELECTED_MODEL = "VGG" # "VGG", "NormalCNN"
SELECTED_PERFORMANCE_OPTIMIZATION_TYPES = [] # ["Quantization", "MixedPrecision", "GradientAccumulation"]
SELECTED_ALGORITHM = "FedAvg" # FedAvg, (1-JSD), ClientSize(1-JSD), AccuracyBased, AccuracyBased(1-JSD), SEBW, AccuracyBased_SEBW, FedProx, CAFA
EXPERIMENT_NAME = f"{SELECTED_ALGORITHM}"
RESULTS_FOLDER_NAME = f'EX-{EXPERIMENT_NAME}_M-{SELECTED_MODEL}_FLR-{FL_ROUNDS}_NC-{NUM_CLIENTS}_CLE-{LOCAL_EPOCHS}_DST-{DATASET_DISTRIBUTION_TYPE}'
RESULTS_DIR = os.path.join(RESULTS_SAVE_PATH, RESULTS_FOLDER_NAME)

""" # 🧭 Initialize wandb
wandb.login(key="cca506f824e9db910b7b4a407afc0b36ba655c28")
wandb.init(
    project=f"FL-Test_0",
    config={
        "model": SELECTED_MODEL,
        "fl_rounds": FL_ROUNDS,
        "clients": NUM_CLIENTS,
        "selected_clients": NUM_SELECTED_CLIENTS,
        "local_epochs": LOCAL_EPOCHS,
        "dataset_distribution": DATASET_DISTRIBUTION_TYPE,
        "device": str(DEVICE)
    },
    name=EXPERIMENT_NAME
) """

# Initialize the Global model and Server
if SELECTED_MODEL == "NormalCNN":
    global_model = NormalCNN().to(DEVICE)
else:
    global_model = VGG('VGG19').to(DEVICE)

# Initialize the Server
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

