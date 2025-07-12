import os
import torch
from typing import Optional
from vision3d.hooks import Hook
from vision3d.engine.trainer import Trainer

from vision3d.utils.logging import configure_logger
from vision3d.utils.registry import HOOKS

logger = configure_logger(__name__.split('.')[-1])

@HOOKS.register()
class CheckpointHook(Hook):
    """
    Hook to save model checkpoints at the end of each epoch.
    
    Args:
        output_dir (str): Directory to save checkpoints.
        save_every (int): Save checkpoint every N epochs.
        save_best (bool): Whether to keep track of best model by val loss.
    """

    def __init__(self, output_dir: str, save_every: int = 1, save_best: bool = True):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.save_every = save_every
        self.save_best = save_best
        self.best_val_loss: Optional[float] = None

    def after_train_epoch(self, epoch: int, trainer: Trainer) -> None:
        # Save regular checkpoint
        if (epoch + 1) % self.save_every == 0:
            ckpt_path = os.path.join(self.output_dir, f"epoch_{epoch}.pth")
            self._save_checkpoint(trainer, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

        # Save best model based on validation loss
        if self.save_best and hasattr(trainer, "last_val_loss"):
            val_loss = trainer.last_val_loss
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(self.output_dir, "best.pth")
                self._save_checkpoint(trainer, best_path)
                logger.info(f"Saved new best model (val_loss={val_loss:.4f}) to {best_path}")

    def _save_checkpoint(self, trainer: Trainer, path: str) -> None:
        torch.save({
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "epoch": trainer.max_epochs,
            "scheduler": trainer.scheduler.state_dict() if trainer.scheduler else None,
        }, path)
