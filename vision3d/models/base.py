from abc import ABC, abstractmethod
from torch import nn

class Vision3DModel(ABC, nn.Module):
    """Abstract base class for Vision3D models."""

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Perform a forward pass of the model.
        """
        pass

    @abstractmethod
    def train_step(self, inputs: dict, targets: dict, optimizer):
        """
        Perform a single training step.
        Args:
            inputs: Dictionary of input data.
            targets: Dictionary of target data.
            optimizer: The optimizer used for updating model parameters.
        """
        pass

    @abstractmethod
    def evaluate(self, inputs: dict, targets: dict,):
        """
        Evaluate the model on the given data.
        Args:
            inputs: Dictionary of input data.
            targets: Dictionary of target data.
        """
        pass