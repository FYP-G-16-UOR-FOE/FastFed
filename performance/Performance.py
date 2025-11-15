import zlib


class Performance:
    def __init__(self, device):
        """
        Initialize Performance object with target device.
        """
        self.device = device

    @staticmethod
    def quantize_model_parameters(state_dict, num_bits=8):
        """
        Quantize model parameters to reduce communication overhead.
        This simulates INT8 quantization for model weights.

        Args:
            state_dict: Model state dictionary
            num_bits: Number of bits for quantization (default: 8 for INT8)

        Returns:
            quantized_dict: Dictionary with quantized parameters
            metadata: Quantization metadata (scales, zero_points) for dequantization
        """
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

        return quantized_dict, metadata

    def dequantize_model_parameters(self, quantized_dict, metadata):
        """
        Dequantize model parameters back to float32.

        Args:
            quantized_dict: Dictionary with quantized parameters
            metadata: Quantization metadata

        Returns:
            state_dict: Dequantized state dictionary
        """
        state_dict = {}

        for key, quantized_param in quantized_dict.items():
            meta = metadata[key]

            if 'scale' in meta:  # This parameter was quantized
                scale = meta['scale']
                zero_point = meta['zero_point']

                # Dequantize
                dequantized = (quantized_param.astype(np.float32) - zero_point) * scale
                state_dict[key] = torch.tensor(dequantized, dtype=meta['dtype']).to(self.device)
            else:
                # This parameter was not quantized
                state_dict[key] = torch.tensor(quantized_param, dtype=meta['dtype']).to(self.device)

        return state_dict

    @staticmethod
    def calculate_size_reduction(original_state_dict, quantized_dict):
        """
        Calculate the size reduction achieved by quantization.

        Args:
            original_state_dict: Original float32 state dict
            quantized_dict: Quantized state dict

        Returns:
            original_size_MB: Original size in MB
            quantized_size_MB: Quantized size in MB
            reduction_percent: Reduction percentage
        """
        # Calculate original size
        original_size = 0
        for param in original_state_dict.values():
            original_size += param.element_size() * param.nelement()

        # Calculate quantized size
        quantized_size = 0
        for param in quantized_dict.values():
            if isinstance(param, np.ndarray):
                quantized_size += param.nbytes
            else:
                quantized_size += param.element_size() * param.nelement()

        original_size_MB = original_size / (1024 * 1024)
        quantized_size_MB = quantized_size / (1024 * 1024)
        reduction_percent = ((original_size_MB - quantized_size_MB) / original_size_MB) * 100

        return original_size_MB, quantized_size_MB, reduction_percent

    @staticmethod
    def serialize_quantized_model(quantized_dict, metadata):
        """
        Serialize quantized model for transmission.
        This simulates the actual bytes that would be sent over the network.

        Returns:
            Serialized bytes object
        """
        buffer = io.BytesIO()
        pickle.dump({'quantized': quantized_dict, 'metadata': metadata}, buffer)
        return buffer.getvalue()

    @staticmethod
    def deserialize_quantized_model(serialized_data):
        """
        Deserialize quantized model received from network.

        Returns:
            quantized_dict, metadata
        """
        buffer = io.BytesIO(serialized_data)
        data = pickle.load(buffer)
        return data['quantized'], data['metadata']

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

    @staticmethod
    def calculate_precision_reduction(original_state_dict, reduced_state_dict):
        """
        Calculate model size and precision reduction achieved.

        Args:
            original_state_dict: Original float32 model
            reduced_state_dict: Reduced precision model

        Returns:
            original_size_MB, reduced_size_MB, reduction_percent
        """
        def tensor_size_in_bytes(tensor):
            return tensor.element_size() * tensor.nelement()

        original_size = sum(tensor_size_in_bytes(p) for p in original_state_dict.values())
        reduced_size = sum(tensor_size_in_bytes(p) for p in reduced_state_dict.values())

        original_size_MB = original_size / (1024 * 1024)
        reduced_size_MB = reduced_size / (1024 * 1024)
        reduction_percent = ((original_size_MB - reduced_size_MB) / original_size_MB) * 100

        return original_size_MB, reduced_size_MB, reduction_percent

    @staticmethod
    def serialize_data(data):
        """
        Serializes a Python object (e.g., model parameters or metrics)
        into compressed bytes for network transmission.
        """
        serialized = pickle.dumps(data)          # Convert object -> bytes
        compressed = zlib.compress(serialized)   # Compress to reduce size
        return compressed

    @staticmethod
    def deserialize_data(byte_data):
        """
        Deserializes compressed byte data back into a Python object.
        """
        decompressed = zlib.decompress(byte_data)
        data = pickle.loads(decompressed)
        return data


# Example usage:
# 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# perf = Performance(device)
# global_model = VGG('VGG19').to(device)
# quantized_dict, metadata = Performance.quantize_model_parameters(global_model.state_dict(), num_bits=8)
# dequantized_dict = perf.dequantize_model_parameters(quantized_dict, metadata)
# reduced_model_dict = Performance.reduce_model_precision(global_model.state_dict(), decimal_places=4)