import torch
from typing import Any, Union
import random
import numpy as np

def to_device(data: Any, device: Union[str, torch.device]) -> Any:
    """
    Recursively move tensors or collections of tensors to the specified device.

    Args:
        data: A torch.Tensor, list/tuple/dict of tensors, or nested combination thereof.
        device: The device to move the data to ('cpu', 'cuda', or torch.device).

    Returns:
        The input data moved to the specified device with the same structure.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    else:
        return data

def seed_everything(seed: int) -> None:
    """
    Seed everything for reproducibility.
    
    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (might slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False