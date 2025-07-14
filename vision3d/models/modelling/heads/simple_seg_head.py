import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Union
from vision3d.utils.registry import MODELS

@MODELS.register()
class SimpleSegHead(nn.Module):
    """
    Simple segmentation head for predicting a segmentation mask.
    """
    def __init__(self, in_channels: int, 
                 hidden_dim: int = 256,
                 out_channels: int = 1,
                 target_hw: tuple =None,
                 ):
        """
        Args:
            in_channels: Number of input channels (from layer1).
            out_channels: Number of output channels (default is 1 for binary mask).
        """
        super(SimpleSegHead, self).__init__()
        self.target_hw = target_hw  
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)  
        self.prob_converter = nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)

    def forward(self, x: Union[Tensor, dict]) -> Tensor:
        """
        Forward pass to predict a segmentation mask.

        Args:
            x: Input tensor of shape (B, C, H, W) or a dict with keys layer1, ..., layer4.

        Returns:
            Binary mask of shape (B, out_channels, H, W).
        """
        x = self._validate_input(x)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        pred_prob = self.prob_converter(x)

        # Bilinearly interpolate to target size
        if self.target_hw is not None:
            pred_prob = F.interpolate(pred_prob, size=self.target_hw, mode='bilinear', align_corners=False)
        
        output = dict(
            pred_mask=pred_prob
        )
        return output
    
    def _validate_input(self, x: Tensor) -> Tensor:
        if isinstance(x, dict):
            if 'layer1' not in x:
                raise ValueError("Input dict must contain 'layer1' key.")
            x = x['layer1']

        if x.dim() != 4:
            raise ValueError(f"Expected input of shape (B, C, H, W), but got {x.shape}")
        return x