import datetime
import json
import os
import random
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch

from vision3d.utils.logging import configure_logger

logger = configure_logger("utils.misc")


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_serializable(obj):
    """Recursively convert non-serializable types to serializable ones."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}  # Recursively process dicts
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]  # Recursively process lists
    else:
        return obj


def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model" in checkpoint and "epoch" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        epoch = checkpoint.get("epoch", "<unknown>")
        logger.info(f"Loaded checkpoint from epoch {epoch}")
    else:
        raise ValueError("Checkpoint format not recognized. Expected 'model' and 'epoch' keys in checkpoint.")

    return model.to(device), epoch


def save_results(results, output_path, epoch_num, split):
    """Save evaluation results to JSON file."""
    result_data = {
        "checkpoint": epoch_num,
        "split": split,
        "timestamp": datetime.datetime.now().isoformat(),
        "metrics": convert_to_serializable(results),
    }
    output_path = Path(output_path) / f"epoch_{epoch_num}_{split}_results.json"

    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def convert_to_onnx(model, checkpoint_path, device, output_path, input_shape):
    """
    Convert a PyTorch model checkpoint to ONNX format and save it.

    Args:
        model: The PyTorch model instance.
        checkpoint_path: Path to the PyTorch checkpoint.
        device: Device to load the model on ('cpu' or 'cuda').
        output_path: Path to save the ONNX model.
        input_shape: Shape of the input tensor (e.g., (1, 3, 224, 224)).
    """
    # Load the checkpoint
    model, _ = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    # Create a dummy input tensor with the specified shape
    if isinstance(input_shape, list):
        dummy_input = [torch.randn(shape).to(device) for shape in input_shape]
    else:
        dummy_input = torch.randn(input_shape).to(device)

    # Convert the model to ONNX format
    onnx_path = Path(output_path)

    torch.onnx.export(
        model,
        tuple(dummy_input),
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["rgb", "pc"],
        output_names=["outputs"],
        dynamic_axes={
            "rgb": {0: "batch_size"},
            "pc": {0: "batch_size"},
            "outputs": {0: "batch_size"},
        },
    )
    logger.info(f"ONNX model saved to: {onnx_path}")
