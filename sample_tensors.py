import json
import base64
import struct
import numpy as np
import torch as t
from typing import Any


def compact_tensor_to_b64(tensor: t.Tensor):
    """
    Optimized for PyTorch tensors.
    """
    # 1. Detach and move to CPU (if not already)
    if tensor.is_cuda:
        tensor = tensor.detach().cpu()
    else:
        tensor = tensor.detach()

    # 2. Convert to NumPy
    # Note: If the tensor is not contiguous (e.g. sliced), call .contiguous()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    arr = tensor.numpy()

    # 3. Flatten and Pack
    raw_bytes = arr.flatten().tobytes()

    # 4. Base64
    b64_str = base64.b64encode(raw_bytes).decode("ascii")

    return {
        "shape": list(tensor.shape),
        "dtype": str(arr.dtype),  # e.g. 'float32'
        "data": b64_str
    }


def b64_to_tensor(obj, as_numpy=True):
    """
    Decode a base64-encoded tensor object back into a NumPy array.
    :param obj: The dict from compact_tensor_to_b64()
    :param as_numpy: Whether to return a NumPy array or as a `torch.Tensor`
    """
    raw_bytes = base64.b64decode(obj["data"])
    shape = tuple(obj["shape"])

    # Robust dtype handling
    dtype_str = obj.get("dtype", "float32")

    # Map struct codes to numpy dtypes if necessary (for backward compat)
    struct_to_np = {"f": "float32", "d": "float64", "e": "float16"}
    if dtype_str in struct_to_np:
        dtype_str = struct_to_np[dtype_str]

    # np.frombuffer is zero-copy and extremely fast
    # It returns a 1D array sharing memory with the bytes object
    flat = np.frombuffer(raw_bytes, dtype=dtype_str)

    # Reshape back to original dimensions
    arr = flat.reshape(shape)

    if as_numpy:
        return arr
    else:
        return t.from_numpy(arr.copy())


def safe_convert_tensor(tensor: t.Tensor) -> Any:
    if isinstance(tensor, t.Tensor):
        if tensor.numel() == 1:
            return tensor.item()
        else:
            return tensor.tolist()
    else:
        return tensor



