import sys

from gRPC import ServergRPC_pb2, ServergRPC_pb2_grpc
from models.NormalCNN import NormalCNN
from models.VGG import VGG
from server.Server import Server


class ServerServicer(ServergRPC_pb2_grpc.ServerServiceServicer):
    def __init__(self, config):
        # initialize server object used for FL logic
        if config["model"] == "NormalCNN":
            global_model = NormalCNN().to(config["device"])
        else:
            global_model = VGG("VGG19").to(config["device"])

        self.server = Server(
            global_model=global_model,
            device=config["device"],
            model_type=config["model"],
            fl_rounds=config["fl_rounds"],
            local_epochs=config["local_epochs"],
            num_clients=config["num_clients"],
            num_selected_clients=config["num_selected_clients"],
            results_save_dir=config["results_dir"],
            selected_algorithm=config["algorithm"],
            is_use_quantization=config["is_use_quantization"],
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