"""Core trainer class for Vision3D models."""

from typing import Callable, List, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from vision3d.engine.evaluator import Evaluator
from vision3d.hooks import Hook
from vision3d.models.base import Vision3DModel
from vision3d.utils.logging import configure_logger
from vision3d.utils.misc import to_device
from vision3d.utils.wandb import WandbLogger

logger = configure_logger(__name__.split(".")[-1])


class Trainer:
    def __init__(
        self,
        model: Vision3DModel,
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        optimizer: Optimizer = None,
        scheduler: LRScheduler = None,
        device: Union[str, torch.device] = "cuda",
        max_epochs: int = 100,
        hooks: List[Hook] = None,
        validate_every: int = 1,
        evaluator: Evaluator = None,
        wandb_logger: WandbLogger = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = max_epochs
        self.hooks = hooks or []
        self.validate_every = validate_every
        self.evaluator = evaluator
        self.wandb_logger = wandb_logger

        self.current_epoch = 0
        self.last_train_loss = None
        self.last_val_step_loss = None
        self.last_val_metrics = None

        # Watch model if wandb is enabled
        if self.wandb_logger and self.wandb_logger.is_enabled():
            self.wandb_logger.watch_model(self.model)

    def run(self) -> None:
        for epoch in range(self.max_epochs):
            self._call_hooks("before_train_epoch", epoch)
            self._train_one_epoch(epoch)
            self._call_hooks("after_train_epoch", epoch)

            if self.val_loader and ((epoch + 1) % self.validate_every == 0):
                self._call_hooks("before_val_epoch", epoch)
                self._validate(epoch)
                self._call_hooks("after_val_epoch", epoch)

            if self.scheduler:
                self.scheduler.step()

                # Log learning rate to wandb
                if self.wandb_logger and self.wandb_logger.is_enabled():
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.wandb_logger.log_metrics(
                        {"learning_rate": current_lr}, step=epoch, commit=False
                    )

            self.current_epoch = epoch

    def _train_one_epoch(self, epoch: int) -> None:
        self.model.train()

        for step, (inputs, targets) in enumerate(self.train_loader):
            inputs = {k: to_device(v, self.device) for k, v in inputs.items()}
            targets = {k: to_device(v, self.device) for k, v in targets.items()}

            self._call_hooks("before_train_step", step)
            loss = self.model.train_step(inputs, targets, self.optimizer)
            # used to store the last loss for hook callbacks
            self.last_train_loss = loss
            self._call_hooks("after_train_step", step)

    def _validate(self, epoch: int) -> None:
        self.model.eval()

        # Reset evaluator if available
        if self.evaluator is not None:
            self.evaluator.reset()

        with torch.no_grad():
            for step, (inputs, targets) in enumerate(self.val_loader):
                inputs = {k: to_device(v, self.device) for k, v in inputs.items()}
                targets = {k: to_device(v, self.device) for k, v in targets.items()}

                # Get model outputs and loss
                outputs, loss = self.model.evaluate(inputs, targets)

                # Update evaluator with the same outputs (no double inference!)
                if self.evaluator is not None:
                    self.evaluator.update(outputs, targets)

                self.last_val_step_loss = loss
                self._call_hooks("after_val_step", step)

        # Compute final metrics after all validation steps
        if self.evaluator is not None:
            self.last_val_metrics = self.evaluator.compute()
            self._log_validation_metrics(epoch)

    def _log_validation_metrics(self, epoch: int) -> None:
        """Log validation metrics."""
        if self.last_val_metrics is None:
            return

        logger.info(f"[Epoch {epoch}] VAL Metrics:")
        for metric_name, metric_values in self.last_val_metrics.items():
            if isinstance(metric_values, dict):
                for sub_metric, value in metric_values.items():
                    logger.info(f"  {metric_name}_{sub_metric}: {value:.4f}")
            else:
                logger.info(f"  {metric_name}: {metric_values:.4f}")

        # Log validation metrics to wandb
        if self.wandb_logger and self.wandb_logger.is_enabled():
            self.wandb_logger.log_metrics(
                metrics=self.last_val_metrics, step=epoch, prefix="val_metrics", commit=False
            )

    def _call_hooks(self, method_name: str, index: int) -> None:
        for hook in self.hooks:
            method = getattr(hook, method_name, None)
            if callable(method):
                method(index, self)

    def evaluate_on_dataloader(self, dataloader: DataLoader) -> dict:
        """
        Standalone evaluation method for any dataloader.

        Args:
            dataloader: DataLoader to evaluate on

        Returns:
            Dictionary of computed metrics
        """
        if self.evaluator is None:
            raise ValueError("No evaluator configured for this trainer")

        self.model.eval()
        self.evaluator.reset()

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = {k: to_device(v, self.device) for k, v in inputs.items()}
                targets = {k: to_device(v, self.device) for k, v in targets.items()}

                outputs, _ = self.model.evaluate(inputs, targets)
                self.evaluator.update(outputs, targets)

        return self.evaluator.compute()

    def get_last_metrics(self) -> Optional[dict]:
        """Get the last computed validation metrics."""
        return self.last_val_metrics
