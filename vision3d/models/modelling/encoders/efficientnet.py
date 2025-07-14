from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision3d.utils.registry import MODELS

from .rgb_encoder import RGBEncoderBase


@MODELS.register()
class EfficientNetEncoder(RGBEncoderBase):
    """
    EfficientNet-based RGB encoder for 3D vision tasks.
    Note: for usage as input to Spatially Aware Transformer it needs to be extended to output 4 scales of features.

    Supports EfficientNet-B0 to EfficientNet-B7 architectures.
    """

    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        pretrained: bool = True,
        freeze: bool = False,
        multiscale: bool = False,
        output_dim: Optional[int] = None,
    ):
        """
        Initialize EfficientNet encoder.

        Args:
            model_name: EfficientNet variant (e.g., "efficientnet-b0", "efficientnet-b4")
            pretrained: Whether to use pretrained weights
            freeze: Whether to freeze the encoder parameters
            multiscale: Whether to return multiscale features
            output_dim: Optional output dimension for feature projection
        """
        self.model_name = model_name
        super().__init__(pretrained, freeze, multiscale, output_dim)

    def _build_encoder(self):
        """Build the EfficientNet encoder."""
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for EfficientNet. Install with: pip install timm")

        # Load pretrained EfficientNet
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            features_only=True,
            out_indices=(2, 3, 4) if self.multiscale else (4,),
        )

        # Get feature info
        feature_info = self.backbone.feature_info

        if self.multiscale:
            self.output_channels = {
                "layer2": feature_info[0]["num_chs"],
                "layer3": feature_info[1]["num_chs"],
                "layer4": feature_info[2]["num_chs"],
            }
        else:
            self.output_channels = feature_info[-1]["num_chs"]

        # Add projection layer if output_dim is specified
        if self.output_dim is not None:
            if self.multiscale:
                self.projections = nn.ModuleDict(
                    {
                        "layer2": nn.Conv2d(self.output_channels["layer2"], self.output_dim, 1),
                        "layer3": nn.Conv2d(self.output_channels["layer3"], self.output_dim, 1),
                        "layer4": nn.Conv2d(self.output_channels["layer4"], self.output_dim, 1),
                    }
                )
                self.output_channels = {
                    "layer2": self.output_dim,
                    "layer3": self.output_dim,
                    "layer4": self.output_dim,
                }
            else:
                self.projection = nn.Conv2d(self.output_channels, self.output_dim, 1)
                self.output_channels = self.output_dim

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through EfficientNet encoder.

        Args:
            x: Input RGB images of shape (B, 3, H, W)

        Returns:
            Features from the backbone
        """
        features = self.backbone(x)

        if self.multiscale:
            feature_dict = {
                "layer2": features[0],
                "layer3": features[1],
                "layer4": features[2],
            }

            # Apply projections if specified
            if self.output_dim is not None:
                feature_dict = {key: self.projections[key](feat) for key, feat in feature_dict.items()}

            return feature_dict
        else:
            output = features[0]

            # Apply projection if specified
            if self.output_dim is not None:
                output = self.projection(output)

            return output
