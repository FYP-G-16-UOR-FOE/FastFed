import pickle
import zlib

import numpy as np
import torch


class Performance:

    @staticmethod
    def quantize_model_parameters(state_dict, num_bits=8):
        quantized_dict = {}
        metadata = {}

        for key, param in state_dict.items():
            if param.dtype == torch.float32 or param.dtype == torch.float16:
                # Calculate scale and zero point for quantization
                param_np = param.cpu().numpy()
                min_val = param_np.min()
                max_val = param_np.max()

                # Calculate quantization parameters
                qmin = 0
                qmax = 2**num_bits - 1

                scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1.0
                zero_point = qmin - min_val / scale if scale != 0 else 0

                # Quantize
                quantized = np.round(param_np / scale + zero_point)
                quantized = np.clip(quantized, qmin, qmax).astype(np.uint8)

                # Store quantized values and metadata
                quantized_dict[key] = quantized
                metadata[key] = {'scale': scale, 'zero_point': zero_point, 'shape': param.shape, 'dtype': param.dtype}
            else:
                # For non-float parameters, keep as is
                quantized_dict[key] = param.cpu().numpy()
                metadata[key] = {'quantized': False, 'shape': param.shape, 'dtype': param.dtype}

        return {"quantized_dict":quantized_dict, "metadata":metadata}

    def dequantize_model_parameters(param):

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        quantized_dict = param["quantized_dict"]
        metadata = param["metadata"]

        state_dict = {}

        for key, quantized_param in quantized_dict.items():
            meta = metadata[key]

            if 'scale' in meta:  # This parameter was quantized
                scale = meta['scale']
                zero_point = meta['zero_point']

                # Dequantize
                dequantized = (quantized_param.astype(np.float32) - zero_point) * scale
                state_dict[key] = torch.tensor(dequantized, dtype=meta['dtype']).to(DEVICE)
            else:
                # This parameter was not quantized
                state_dict[key] = torch.tensor(quantized_param, dtype=meta['dtype']).to(DEVICE)

        return state_dict

    @staticmethod
    def reduce_model_precision(state_dict, decimal_places=3):
        """
        Reduce model size by approximating float values to fewer decimal points.

        Args:
            state_dict: Model state dictionary
            decimal_places: Number of decimal places to keep (default: 3)

        Returns:
            reduced_dict: Dictionary with reduced precision tensors
        """
        reduced_dict = {}
        for key, param in state_dict.items():
            if torch.is_floating_point(param):
                # Round float parameters to specified precision
                reduced_param = torch.round(param * (10 ** decimal_places)) / (10 ** decimal_places)
                reduced_dict[key] = reduced_param
            else:
                # Keep non-float parameters as is
                reduced_dict[key] = param.clone()
        return reduced_dict