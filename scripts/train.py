import argparse
import os
import torch
from torch.utils.data import DataLoader
from vision3d.utils.config import load_config
from vision3d.engine.trainer import Trainer
from vision3d.utils.misc import seed_everything
from vision3d.utils.build import build_model, build_dataset, build_hook, build_optimizer, build_scheduler, build_utils
from vision3d.datasets.detection3d_dataset import collate_fn
from pprint import pprint
from vision3d.utils.wandb import WandbLogger
import datetime

from vision3d.utils.logging import configure_logger
logger = configure_logger(__name__.split(".")[-1])

def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D vision model")
    parser.add_argument("--config", required=True, help="Path to the config file")
    parser.add_argument("--save_dir", required=True, help="Directory to save logs and checkpoints")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    return parser.parse_args()

def check_core_components(cfg):
    required_keys = ["model", "train", "optimizer"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required configuration key: {key}")

    if "scheduler" in cfg and not isinstance(cfg["scheduler"], dict):
        raise ValueError("Scheduler configuration must be a dictionary")

    if "hooks" in cfg and not isinstance(cfg["hooks"], list):
        raise ValueError("Hooks configuration must be a list of dictionaries")

def main():
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    args = parse_args()
    cfg = load_config(args.config, return_edict=True)

    check_core_components(cfg)

    # run name is a name of config + timestamp
    RUN_NAME = args.config.split("/")[-1].split(".")[0] + "_" + timestamp
    save_dir = args.save_dir + f"/{RUN_NAME}"
    os.makedirs(save_dir, exist_ok=True)

    if "seed" in cfg:
        seed_everything(cfg.seed)   
        logger.info(f"Set random seed to {cfg.seed}")

    # Initialize wandb logger
    wandb_logger = None
    if args.use_wandb or cfg.get("use_wandb", False):
        wandb_logger = WandbLogger(
            enabled=True,
            project_name=cfg.get("wandb_project_name", "vision3d"),
            run_name=RUN_NAME,
        )
        logger.info("WandB logging enabled")

    # Build components
    train_dataset = build_dataset(cfg.train_dataset)
    val_dataset = build_dataset(cfg.val_dataset) if "val_dataset" in cfg else None


    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.val.batch_size,
            shuffle=False,
            num_workers=cfg.val.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    model = build_model(cfg.model)
    optimizer = build_optimizer(cfg.optimizer, model.parameters())
    scheduler = build_scheduler(cfg.scheduler, optimizer) if "scheduler" in cfg else None
    hooks = []
    for h in cfg.get("hooks", []):
        
        hook = build_hook(h)
         # Inject save_dir into hooks that support it (like CheckpointHook)
        if hasattr(hook, 'set_save_dir'):
            hook.set_save_dir(save_dir)
            logger.info(f"Injected save_dir into {hook.__class__.__name__}")
        
        # Inject wandb_logger into hooks that support it
        if hasattr(hook, 'set_wandb_logger'):
            hook.set_wandb_logger(wandb_logger)
            logger.info(f"Injected wandb_logger into {hook.__class__.__name__}")
        hooks.append(hook)

    evaluator = build_utils(cfg.evaluator) if "evaluator" in cfg else None
    

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=cfg.get("device", "cuda"),
        max_epochs=cfg.train.epochs,
        hooks=hooks,
        evaluator=evaluator,
        wandb_logger=wandb_logger
    )

    logger.info(f"Training configuration: {pprint(cfg, indent=4)}")
    logger.info(f"Start training ...")

    try:
        trainer.run()
    finally:
        # Clean up wandb
        if wandb_logger:
            wandb_logger.finish()
            logger.info("WandB logging finished")



if __name__ == "__main__":
    main()

"""usage example:
python vision3d/scripts/train.py --config ./vision3d/configs/dummy.py --save_dir ./logs/dummy
python scripts/train.py --config /home/hao1rng/sec_proj/vision3d/configs/dummy_detr3d.py.py --save_dir ./logs/dummy
python scripts/train.py --config /home/hao1rng/sec_proj/vision3d/configs/mono_detr3d.py --save_dir ./logs/mono_detr3d
"""