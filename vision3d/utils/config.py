"""Utilities to handle configuration files."""

import importlib.util
import os
from typing import Union    

from easydict import EasyDict as edict

def to_edict(obj):
    if isinstance(obj, dict):
        return edict({k: to_edict(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [to_edict(item) for item in obj]
    else:
        return obj

def load_config(file_path: str, return_edict: bool=False) -> Union[dict, edict]:
    # Resolve the absolute path and derive a unique module name from the file name.
    file_path = os.path.abspath(file_path)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Create a module specification based on the file location.
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise FileNotFoundError(f"Could not load spec for {file_path}")
    
    # Create the module based on the specification.
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"No loader found for {file_path}")
    
    # Execute the module in its own namespace.
    spec.loader.exec_module(module)
    
    # Extract public attributes (usually your config dictionaries).
    config_dict = {k: v for k, v in vars(module).items() if not k.startswith("__")}

    if return_edict:
        config_dict = to_edict(config_dict)

    return config_dict