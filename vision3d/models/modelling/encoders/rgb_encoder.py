from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import torch
import torch.nn as nn


class RGBEncoderBase(nn.Module, ABC):
    """
    Base class for RGB encoders used in 3D vision tasks.

    This class provides the interface for encoding RGB images into feature representations
    that can be used for 3D detection, segmentation, or other vision tasks.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze: bool = False,
        multiscale: bool = False,
        output_dim: Optional[int] = None,
    ):
        """
        Initialize the RGB encoder.

        Args:
            pretrained: Whether to use pretrained weights
            freeze: Whether to freeze the encoder parameters
            multiscale: Whether to return multiscale features
            output_dim: Optional output dimension for feature projection
        """
        super().__init__()
        self.pretrained = pretrained
        self.freeze = freeze
        self.multiscale = multiscale
        self.output_dim = output_dim

        self._build_encoder()

        if self.freeze:
            self._freeze_parameters()

    @abstractmethod
    def _build_encoder(self):
        """Build the encoder architecture."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the encoder.

        Args:
            x: Input RGB images of shape (B, 3, H, W)

        Returns:
            If multiscale=False: Feature tensor of shape (B, C, H', W')
            If multiscale=True: Dict with keys ['layer1', 'layer2', 'layer3', ...] containing
                               features from different spatial scales
        """
        pass

    def _freeze_parameters(self):
        """Freeze all parameters in the encoder."""
        for param in self.parameters():
            param.requires_grad = False
