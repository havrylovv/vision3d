import torch
from torch import nn
from vision3d.utils.registry import LOSSES

@LOSSES.register()
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, reduction=reduction,)

    def forward(self, input, target):
        return self.loss_fn(input, target)