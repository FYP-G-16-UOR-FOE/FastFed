import copy
import io

import torch

from models.VGG import VGG


# -------------------------------
# Helper: serialize size function
# -------------------------------
def serialized_size_of_state_dict(state_dict):
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return len(buf.getvalue())


# -------------------------------
# 1. Load Global Model (VGG19)
# -------------------------------
print("Loading VGG19 global model...")
global_model = VGG("VGG19")
global_state = global_model.state_dict()

# Size of global model
global_size = serialized_size_of_state_dict(global_state)
print(f"Global Model Size: {global_size/1024/1024:.2f} MB")


# -------------------------------
# 2. Simulate Client Model
# -------------------------------
print("Creating simulated client model...")
client_model = copy.deepcopy(global_model)

# Apply random update to simulate training
with torch.no_grad():
    for k, v in client_model.state_dict().items():
        if v.dtype.is_floating_point:
            client_model.state_dict()[k] = v + 0.01 * torch.randn_like(v)
        else:
            # integer/buffer tensors remain unchanged
            client_model.state_dict()[k] = v.clone()

client_state = client_model.state_dict()

# Size of client model
client_size = serialized_size_of_state_dict(client_state)
print(f"Simulated Client Model Size: {client_size/1024/1024:.2f} MB")


# -------------------------------
# 3. Compute Delta = client - global
# -------------------------------
print("Computing delta update...")
delta_state = {}
for k in global_state.keys():
    delta_state[k] = client_state[k] - global_state[k]

# Size of delta
delta_size = serialized_size_of_state_dict(delta_state)
print(f"Delta Update Size: {delta_size/1024/1024:.2f} MB")


# -------------------------------
# 4. Reconstruct Model from Delta
# -------------------------------
print("Reconstructing new global model from delta...")
reconstructed_state = {}
for k in global_state.keys():
    reconstructed_state[k] = global_state[k] + delta_state[k]

# Load into a fresh model
reconstructed_model = VGG("VGG19")
reconstructed_model.load_state_dict(reconstructed_state)


# -------------------------------
# 5. Compare reconstructed vs simulated client model
# -------------------------------
print("Comparing reconstruction correctness...")

def models_equal(m1, m2):
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        if not torch.allclose(v1, v2, atol=1e-6):
            print(f"Mismatch at layer: {k1}")
            return False
    return True


same = models_equal(reconstructed_model, client_model)

print("---------------------------------------")
print("Delta Update Verification Result:")
print(f"Reconstructed Model == Client Model ? {same}")
print("---------------------------------------")