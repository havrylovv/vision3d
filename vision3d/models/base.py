from abc import ABC, abstractmethod
from torch import nn

class Vision3DModel(ABC, nn.Module):
    """Abstract base class for Vision3D models."""

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Perform a forward pass of the model.
        This method must be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def train_step(self, data, optimizer):
        """
        Perform a single training step.
        Args:
            data: The input data for training.
            optimizer: The optimizer used for updating model parameters.
        """
        pass

    @abstractmethod
    def evaluate(self, data):
        """
        Evaluate the model on the given data.
        Args:
            data: The input data for evaluation.
        """
        pass