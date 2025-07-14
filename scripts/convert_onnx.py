"""Script to convert a PyTorch model checkpoint to ONNX format."""

import argparse

from vision3d.utils.build import build_model
from vision3d.utils.config import load_config
from vision3d.utils.misc import convert_to_onnx


def check_core_components(cfg):
    """Check if configuration has required components for evaluation."""
    required_keys = ["model", "input_shape"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required configuration key: {key}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to ONNX format.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the config file that has information about the model.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the PyTorch checkpoint.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the ONNX model.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load the model on ('cpu' or 'cuda').")
    return parser.parse_args()


if __name__ == "__main__":
    # load cfg
    args = parse_args()
    cfg = load_config(args.config, return_edict=True)
    check_core_components(cfg)

    # build model
    model = build_model(cfg.model)

    # convert to ONNX
    convert_to_onnx(model, args.checkpoint, args.device, args.output_path, cfg.input_shape)
