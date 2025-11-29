import sys

from client.Client import Client
from gRPC import ClientgRPC_pb2, ClientgRPC_pb2_grpc
from performance.Performance import Performance
from utils.Serializer import Serializer


class ClientServicer(ClientgRPC_pb2_grpc.ClientServiceServicer):
    def __init__(self, client: Client):
        self.client = client

    def ReceiveGlobalModel(self, messages, context):
        try:
            client_id: int = None
            model_bytes: bytes = b""
            for i, message in enumerate(messages):
                if i == 0 and client_id is None:
                    client_id = message.client_id
                model_bytes += message.global_model
            print(f"[ClientServicer] ReceiveGlobalModel for client {client_id}")

            state_dict = Serializer.deserialize(model_bytes)
            if self.client.fl_performance["is_use_quantization"]:
                state_dict = Performance.dequantize_model_parameters(state_dict)

            self.client.receive_global_model(state_dict)
            return ClientgRPC_pb2.StatusResponse(status="OK")
        except Exception as e:
            print("[Client] Error applying global model:", e, file=sys.stderr)
            raise e
        
    def StartLocalTraining(self, request, context):
        client_id = request.client_id
        print(f"[ClientServicer] StartLocalTraining for client {client_id}")
        try:
            self.client.start_client_local_training()
            return ClientgRPC_pb2.StatusResponse(status="OK")
        except Exception as e:
            print("[Client] Error during local training:", e, file=sys.stderr)
            raise e
        
    def GetClientsTrainedModel(self, request, context):
        client_id = request.client_id
        try:
            client_model_params, training_time = self.client.get_client_updates()
            if self.client.fl_performance["is_use_quantization"]:
                client_model_params = Performance.quantize_model_parameters(client_model_params)

            model_bytes = Serializer.serialize(client_model_params)

            yield ClientgRPC_pb2.GetClientsTrainedModelResponse(
                client_id=client_id,
                training_time=training_time
            )

            chunk_size = 1*1024*1024  # 1 MB
            for i in range(0, len(model_bytes), chunk_size):
                yield ClientgRPC_pb2.GetClientsTrainedModelResponse(
                    trained_model=model_bytes[i:i+chunk_size]
                )
            
        except Exception as e:
            print("[Client] Error getting trained model:", e, file=sys.stderr)
            raise e
        
    def GetIIDMeasure(self, request, context):
        client_id = request.client_id
        print(f"[ClientServicer] GetIIDMeasure for client {client_id}")
        try:
            iid_measure = self.client.calculate_iid_measure()
            return ClientgRPC_pb2.GetIIDMeasureResponse(
                client_id=client_id,
                iid_measure=iid_measure
            )
        except Exception as e:
            print("[Client] Error calculating IID measure:", e, file=sys.stderr)
            raise e
        
    def GetClassificationIIDMeasure(self, request, context):
        client_id = request.client_id
        print(f"[ClientServicer] GetClassificationIIDMeasure for client {client_id}")
        try:
            client_distribution = self.client.calculate_label_distribution()
            return ClientgRPC_pb2.GetClassificationIIDMeasureResponse(
                client_id=client_id,
                dataset_distribution=client_distribution
            )
        except Exception as e:
            print("[Client] Error calculating IID measure:", e, file=sys.stderr)
            raise e

    def ReceiveModelForAccuracyBasedMeasure(self, messages, context):
        try:
            client_id: int = None
            model_bytes: bytes = b""
            model_type = None
            for i, message in enumerate(messages):
                if i == 0 and client_id is None:
                    client_id = message.client_id
                    model_type = message.model_type
                model_bytes += message.model
            print(f"[ClientServicer] ReceiveGlobalModel for client {client_id}")

            state_dict = Serializer.deserialize(model_bytes)

            if self.client.fl_performance["is_use_quantization"]:
                state_dict = Performance.dequantize_model_parameters(state_dict)

            weighted_val_acc = self.client.evaluate_model_on_validation_data(state_dict=state_dict, model_type=model_type)
            return ClientgRPC_pb2.AccuracyBasedMeasureResponse(weighted_val_acc=weighted_val_acc)
        except Exception as e:
            print("[Client] Error computing accuracy-based measure:", e, file=sys.stderr)
            raise e