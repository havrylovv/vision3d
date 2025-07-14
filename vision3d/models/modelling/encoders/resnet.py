import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import torchvision.models as models
from vision3d.utils.registry import MODELS
from vision3d.models.modelling.encoders import RGBEncoderBase

@MODELS.register()
class ResNetEncoder(RGBEncoderBase):
    """
    ResNet-based RGB encoder for 3D vision tasks.
    
    Supports ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 architectures.
    """
    
    SUPPORTED_MODELS = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }
    
    def __init__(
        self,
        model_name: Union[int, str] = 50,
        pretrained: bool = True,
        freeze: bool = False,
        multiscale: bool = False,
        output_dim: Optional[int] = None,
    ):
        """
        Initialize ResNet encoder.
        
        Args:
            model_name: ResNet variant (18, 34, 50, 101, 152)
            pretrained: Whether to use pretrained weights
            freeze: Whether to freeze the encoder parameters
            multiscale: Whether to return multiscale features
            output_dim: Optional output dimension for feature projection
        """

        if isinstance(model_name, str):
            model_name = int(model_name)
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported ResNet model: {model_name}. "
                           f"Supported models: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_name = model_name
        super().__init__(pretrained, freeze, multiscale, output_dim)
    
    def _build_encoder(self):
        """Build the ResNet encoder."""
        # Load pretrained ResNet
        model_fn = self.SUPPORTED_MODELS[self.model_name]
        self.backbone = model_fn(pretrained=self.pretrained)
        
        # Remove the final classification layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Extract individual layers for multiscale features
        self.layer1 = nn.Sequential(*list(self.backbone.children())[:5])  # Conv1, BN, ReLU, MaxPool, Layer1
        self.layer2 = self.backbone[5]  # Layer2
        self.layer3 = self.backbone[6]  # Layer3
        self.layer4 = self.backbone[7]  # Layer4
        
        # Determine output channels
        if self.model_name in [18, 34]:
            # BasicBlock
            self.output_channels = {
                'layer1': 64,
                'layer2': 128,
                'layer3': 256,
                'layer4': 512,
            }
        else:
            # Bottleneck
            self.output_channels = {
                'layer1': 256,
                'layer2': 512,
                'layer3': 1024,
                'layer4': 2048,
            }
        
        # Add projection layer if output_dim is specified
        if self.output_dim is not None:
            if self.multiscale:
                self.projections = nn.ModuleDict({
                    'layer1': nn.Conv2d(self.output_channels['layer1'], self.output_dim, 1),
                    'layer2': nn.Conv2d(self.output_channels['layer2'], self.output_dim, 1),
                    'layer3': nn.Conv2d(self.output_channels['layer3'], self.output_dim, 1),
                    'layer4': nn.Conv2d(self.output_channels['layer4'], self.output_dim, 1),
                })
                self.output_channels = {
                    'layer1': self.output_dim,
                    'layer2': self.output_dim,
                    'layer3': self.output_dim,
                    'layer4': self.output_dim,
                }
            else:
                self.projection = nn.Conv2d(self.output_channels['layer4'], self.output_dim, 1)
                self.output_channels = self.output_dim
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through ResNet encoder.
        
        Args:
            x: Input RGB images of shape (B, 3, H, W)
            
        Returns:
            If multiscale=False: Feature tensor of shape (B, C, H/32, W/32)
            If multiscale=True: Dict with keys ['layer1', 'layer2', 'layer3', 'layer4'] containing
                    features of shapes (B, C, H/4, W/4), (B, C, H/8, W/8), 
                    (B, C, H/16, W/16), (B, C, H/32, W/32)
        """
        # Forward through layers
        layer1_out = self.layer1(x)           # (B, C, H/4, W/4)
        layer2_out = self.layer2(layer1_out)  # (B, C, H/8, W/8)
        layer3_out = self.layer3(layer2_out)  # (B, C, H/16, W/16)
        layer4_out = self.layer4(layer3_out)  # (B, C, H/32, W/32)
        
        if self.multiscale:
            # Return multiscale features
            features = {
                'layer1': layer1_out,
                'layer2': layer2_out,
                'layer3': layer3_out,
                'layer4': layer4_out,
            }
            
            # Apply projections if specified
            if self.output_dim is not None:
                features = {
                    key: self.projections[key](feat) 
                    for key, feat in features.items()
                }
            
            return features
        else:
            # Return single scale features
            features = layer4_out
            
            # Apply projection if specified
            if self.output_dim is not None:
                features = self.projection(features)
            
            return features
    
    def get_output_channels(self) -> Union[int, Dict[str, int]]:
        """Get the number of output channels."""
        return self.output_channels
    

def test():
    """Run a simple test of the ResNetEncoder."""
    encoder = ResNetEncoder(model_name=18, pretrained=True, multiscale=True, output_dim=128)
    x = torch.randn(2, 3, 512, 512)  
    features = encoder(x)
    
    print("ResNetEncoder test passed.") 
    print(f"features.keys: {features.keys()}")
    print(f"layer1 shape: {features['layer1'].shape}")
    print(f"layer2 shape: {features['layer2'].shape}")
    print(f"layer3 shape: {features['layer3'].shape}")
    print(f"layer4 shape: {features['layer4'].shape}")

if __name__ == "__main__":
    test()