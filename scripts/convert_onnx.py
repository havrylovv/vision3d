import torch
import argparse
from pathlib import Path
from vision3d.utils.misc import load_checkpoint
from vision3d.utils.build import build_model
from vision3d.utils.config import load_config


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
        dynamic_axes={"rgb": {0: "batch_size"}, 
                      "pc": {0: "batch_size"},  
                      "outputs": {0: "batch_size"},
                    }
    )
    print(f"ONNX model saved to: {onnx_path}")

def check_core_components(cfg):
    """Check if configuration has required components for evaluation."""
    required_keys = ["model", "input_shape"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required configuration key: {key}")
        

def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to ONNX format.")
    parser.add_argument("--config", required=True, help="Path to the config file that has information about the model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the PyTorch checkpoint.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the ONNX model.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load the model on ('cpu' or 'cuda').")
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    cfg = load_config(args.config, return_edict=True)

    check_core_components(cfg)
    
    model = build_model(cfg.model)

    convert_to_onnx(model, args.checkpoint, args.device, args.output_path, cfg.input_shape)
