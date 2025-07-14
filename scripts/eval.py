"""Script to evaluate a trained model using a specified configuration and checkpoint."""

import argparse
import datetime
import os
from pprint import pprint

from torch.utils.data import DataLoader

from vision3d.datasets.detection3d_dataset import collate_fn
from vision3d.engine.trainer import Trainer
from vision3d.utils.build import build_dataset, build_model, build_utils
from vision3d.utils.config import load_config
from vision3d.utils.logging import configure_logger
from vision3d.utils.misc import (
    convert_to_serializable,
    load_checkpoint,
    save_results,
    seed_everything,
)

logger = configure_logger(__name__.split(".")[-1])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a 3D vision model")
    parser.add_argument("--config", required=True, help="Path to the config file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--eval_dir", required=True, help="Directory to save evaluation results")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate on",
    )
    return parser.parse_args()


def check_core_components(cfg):
    """Check if configuration has required components for evaluation."""
    required_keys = ["model", "test_dataset", "evaluator"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required configuration key: {key}")


def main():
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    args = parse_args()
    cfg = load_config(args.config, return_edict=True)

    check_core_components(cfg)

    # Create evaluation directory
    RUN_NAME = args.config.split("/")[-1].split(".")[0] + "_" + timestamp
    save_dir = args.eval_dir + f"/{RUN_NAME}"
    os.makedirs(save_dir, exist_ok=True)

    # Set random seed if specified
    if "seed" in cfg:
        seed_everything(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")

    # Build dataset based on split
    if args.split == "train":
        dataset_cfg = cfg.train_dataset
    elif args.split == "val":
        dataset_cfg = cfg.val_dataset
    elif args.split == "test":
        dataset_cfg = cfg.get("test_dataset", None)
        if dataset_cfg is None:
            logger.warning("No test_dataset found in config, using val_dataset for evaluation")
            dataset_cfg = cfg.val_dataset

    dataset = build_dataset(dataset_cfg)

    # Create data loader
    eval_dataloader = DataLoader(
        dataset,
        batch_size=cfg.val.batch_size,
        shuffle=False,
        num_workers=cfg.val.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Build model
    model = build_model(cfg.model)
    mode, epoch = load_checkpoint(model, args.checkpoint, cfg.device)
    model = model.to(cfg.device)

    # Build evaluator
    evaluator = build_utils(cfg.evaluator)
    if evaluator is None:
        raise ValueError("No evaluator found in config. Evaluation requires an evaluator.")

    # Create trainer
    trainer = Trainer(
        model=model,
        device=cfg.device,
        evaluator=evaluator,
    )

    # Evaluate on the dataset
    eval_results = trainer.evaluate_on_dataloader(eval_dataloader)

    # Save and display results
    save_results(eval_results, save_dir, epoch, args.split)

    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Evaluation Split: {args.split}")
    logger.info("Metrics:")
    logger.info(pprint(convert_to_serializable(eval_results)))
    logger.info("=" * 50)
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
