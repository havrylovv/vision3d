import argparse
import os
import torch
from torch.utils.data import DataLoader
from vision3d.utils.config import load_config
from vision3d.engine.trainer import Trainer
from vision3d.utils.misc import seed_everything
from vision3d.utils.build import build_model, build_dataset, build_hook, build_optimizer, build_scheduler
from vision3d.datasets.detection3d_dataset import collate_fn
from pprint import pprint

from vision3d.utils.logger import configure_logger
logger = configure_logger(__name__.split(".")[-1])

def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D vision model")
    parser.add_argument("--config", required=True, help="Path to the config file")
    parser.add_argument("--save_dir", required=True, help="Directory to save logs and checkpoints")
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
    args = parse_args()
    cfg = load_config(args.config, return_edict=True)

    check_core_components(cfg)

    os.makedirs(args.save_dir, exist_ok=True)

    if "seed" in cfg:
        seed_everything(cfg.seed)   
        logger.info(f"Set random seed to {cfg.seed}")

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
    hooks = [build_hook(h) for h in cfg.get("hooks", [])]

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=cfg.get("device", "cuda"),
        max_epochs=cfg.train.epochs,
        hooks=hooks,
    )

    logger.info(f"Training configuration: {pprint(cfg, indent=4)}")
    logger.info(f"Start training ...")

    trainer.run()




if __name__ == "__main__":
    main()

"""usage example:
python vision3d/scripts/train.py --config ./vision3d/configs/dummy.py --save_dir ./logs/dummy
python scripts/train.py --config /home/hao1rng/sec_proj/vision3d/configs/dummy_detr3d.py.py --save_dir ./logs/dummy
"""