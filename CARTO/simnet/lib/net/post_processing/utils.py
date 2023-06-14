import torch
import numpy as np


def to_numpy_from_torch(torch_tensor: torch.Tensor, multiplier: float = 100.0):
    numpy_array = np.ascontiguousarray(torch_tensor.float().cpu().numpy())
    if numpy_array.ndim == 3:  # Not batched
        # print(f"not batched {numpy_array.shape = }")
        numpy_array = np.expand_dims(numpy_array, 0)  # Add one dimension
    numpy_array = numpy_array.transpose((0, 2, 3, 1))
    return numpy_array / multiplier


def to_torch_from_numpy(numpy_array: np.ndarray, multiplier: float = 100.0):
    numpy_array = numpy_array.transpose((2, 0, 1))
    numpy_array = numpy_array * multiplier
    torch_tensor = torch.from_numpy(np.ascontiguousarray(numpy_array)).float()
    return torch_tensor
