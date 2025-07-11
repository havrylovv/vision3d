import json
import os
from collections import defaultdict
from typing import Union
from vision3d.hooks import Hook
from vision3d.utils.registry import HOOKS
from vision3d.utils.logger import configure_logger

logger = configure_logger(__name__.split('.')[-1])

@HOOKS.register()
class LossLoggingHook(Hook):
    def __init__(self, 
                 log_interval: int = 100,
                 aggregate_train_loss: bool = True,
                 aggregate_val_loss: bool = True):
        self.log_interval = log_interval
        self.aggregate_train_loss = aggregate_train_loss
        self.aggregate_val_loss = aggregate_val_loss

        self.train_loss_buffer = []
        self.val_loss_buffer = []
        self.train_total_steps = 0
        self.val_total_steps = 0

    def before_train_epoch(self, epoch: int, trainer):
        self.train_total_steps = len(trainer.train_loader)

    def before_val_epoch(self, epoch: int, trainer):
        self.val_total_steps = len(trainer.val_loader)

    def after_train_step(self, step: int, trainer):
        loss = trainer.last_train_loss
        self._append_loss(self.train_loss_buffer, loss)

        if self.aggregate_train_loss and (step + 1) % self.log_interval == 0:
            self._log_loss(trainer.current_epoch, step + 1, self.train_total_steps, "TRAIN", self.train_loss_buffer)
            self.train_loss_buffer.clear()

    def after_train_epoch(self, epoch: int, trainer):
        if self.aggregate_train_loss and self.train_loss_buffer:
            self._log_loss(epoch, None, None, "TRAIN", self.train_loss_buffer)
            self.train_loss_buffer.clear()

    def after_val_step(self, step: int, trainer):
        loss = trainer.last_val_step_loss
        self._append_loss(self.val_loss_buffer, loss)

        if self.aggregate_val_loss and (step + 1) % self.log_interval == 0:
            self._log_loss(trainer.current_epoch, step + 1, self.val_total_steps, "VAL", self.val_loss_buffer)
            self.val_loss_buffer.clear()

    def after_val_epoch(self, epoch: int, trainer):
        if self.aggregate_val_loss and self.val_loss_buffer:
            self._log_loss(epoch, None, None, "VAL", self.val_loss_buffer)
            self.val_loss_buffer.clear()

    def _append_loss(self, buffer, loss):
        if isinstance(loss, dict):
            buffer.append({k: v.item() if hasattr(v, "item") else v for k, v in loss.items()})
        else:
            buffer.append(loss.item() if hasattr(loss, "item") else loss)

    def _log_loss(self, epoch: int, step: Union[int, None], total: Union[int, None], phase: str, buffer):
        if isinstance(buffer[0], dict):
            aggregated = defaultdict(float)
            for loss_dict in buffer:
                for k, v in loss_dict.items():
                    aggregated[k] += v
            for k in aggregated:
                aggregated[k] /= len(buffer)
            loss_str = " - ".join(f"{k}: {v:.4f}" for k, v in aggregated.items())
        else:
            avg_loss = sum(buffer) / len(buffer)
            loss_str = f"loss: {avg_loss:.4f}"

        if step is not None and total is not None:
            logger.info(f"[Epoch {epoch} {step}/{total}] {phase} - {loss_str}")
        else:
            logger.info(f"[Epoch {epoch}] {phase} - {loss_str}")
