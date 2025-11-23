import io

import torch


class Serializer:
    @staticmethod
    def serialize_state_dict(state_dict) -> bytes:
        buf = io.BytesIO()
        # save CPU tensors to avoid CUDA device issues during load
        cpu_state = {k: v.cpu() for k, v in state_dict.items()}
        torch.save(cpu_state, buf)
        return buf.getvalue()

    @staticmethod
    def deserialize_state_dict(b: bytes, map_location=None):
        buf = io.BytesIO(b)
        return torch.load(buf, map_location=map_location)