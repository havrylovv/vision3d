from typing import Optional, List, Callable, Union
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from vision3d.hooks import Hook
from vision3d.utils.misc import to_device
from vision3d.models.base import Vision3DModel

class Trainer:
    def __init__(
        self,
        model: Vision3DModel,  
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        device: Union[str, torch.device] = "cuda",
        max_epochs: int = 100,
        hooks: Optional[List[Hook]] = None,
        validate_every: int = 1,
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

        self.current_epoch = 0

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

        with torch.no_grad():
            for step, (inputs, target) in enumerate(self.val_loader):
                
                inputs = {k: to_device(v, self.device) for k, v in inputs.items()}
                target = {k: to_device(v, self.device) for k, v in target.items()}
                outs, loss = self.model.evaluate(inputs, target)

                self.last_val_step_loss = loss
                self._call_hooks("after_val_step", step)

    def _call_hooks(self, method_name: str, index: int) -> None:
        for hook in self.hooks:
            method = getattr(hook, method_name, None)
            if callable(method):
                method(index, self)
