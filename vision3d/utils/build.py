import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from vision3d.utils.registry import MODELS, LOSSES, DATASETS, HOOKS, Registry

def build_from_cfg(cfg: dict, registry: Registry) -> object:
    return registry.build(cfg)

def build_model(cfg: dict) -> object:
    return build_from_cfg(cfg, MODELS)

def build_loss(cfg) -> object:
    return build_from_cfg(cfg, LOSSES)

def build_dataset(cfg) -> object:
    return build_from_cfg(cfg, DATASETS)

def build_hook(cfg) -> object:
    return build_from_cfg(cfg, HOOKS)

def build_optimizer(cfg: dict, params) -> Optimizer:
    """Builds an optimizer from the given configuration.
    Supports only native PyTorch optimizers. To include custom optimizers, need to extend Registry logic first.
    
    Args:
        cfg (dict): Configuration dictionary containing optimizer type and parameters.
        params (iterable): Parameters to optimize.
    
    Returns:
        Optimizer: An instance of the optimizer.
    """
    opt_cls = getattr(torch.optim, cfg["type"])
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    return opt_cls(params, **kwargs)

def build_scheduler(cfg: dict, optimizer: Optimizer) -> _LRScheduler:
    """Builds a learning rate scheduler from the given configuration.
    Only supports native PyTorch schedulers. To include custom schedulers, need to extend Registry logic first.
    
    Args:
        cfg (dict): Configuration dictionary containing scheduler type and parameters.
        optimizer (Optimizer): The optimizer to attach the scheduler to.
    Returns:
        _LRScheduler: An instance of the learning rate scheduler.
    """
    sched_cls = getattr(torch.optim.lr_scheduler, cfg["type"])
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    return sched_cls(optimizer, **kwargs)