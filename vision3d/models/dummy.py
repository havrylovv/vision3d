import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Dict, Any
from vision3d.models.base import Vision3DModel
from vision3d.utils.build import build_loss
from vision3d.utils.registry import MODELS

@MODELS.register()
class DummyMLPModel(Vision3DModel):
    def __init__(self, input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 criterion: dict,
                 ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.criterion = build_loss(criterion)  

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Assume inputs dict contains "features" key with input tensor
        x = inputs["features"]
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

    def train_step(self, data: dict, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Perform a training step given a batch dictionary.

        Args:
            data: A dict containing inputs and targets, e.g.:
                {
                    "rgb": tensor,
                    "pc": tensor,
                    "mask": tensor,
                    "bbox3d": tensor,
                    ...
                }
            optimizer: optimizer to update parameters

        Returns:
            loss value as a tensor
        """
        inputs = {"rbg": data["rgb"], "pc": data["pc"]}
        targets = {"mask": data["mask"], "bbox3d": data["bbox3d"]}

        # generate some random input and target tensors for demonstration
        inputs = {"features": torch.randn(32, 64).to('cuda')}  # Example input tensor
        targets = torch.randn(32, 10).to('cuda')  # Example target tensor

        optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        return loss

    def evaluate(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # random data for demonstration
        batch = {
            "features": torch.randn(32, 64).to('cuda'),  # Example input tensor
        }
        inputs = {"features": batch["features"]}
        targets = torch.randn(32, 10).to('cuda')  
        
        #targets = batch["targets"]
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        return loss.detach()
