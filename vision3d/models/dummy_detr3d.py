import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from scipy.optimize import linear_sum_assignment
import numpy as np

from vision3d.utils.registry import MODELS
from vision3d.utils.build import build_loss

#@MODELS.register()
class SimpleCNNBackbone(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

@MODELS.register()
class DummyDETR3D(nn.Module):
    def __init__(self, 
                num_queries=100,
                hidden_dim=256,
                backbone_args: dict = {'out_channels': 256},
                transformer_args: dict = {'d_model': 256, 'nhead': 8, 'num_encoder_layers': 6, 'num_decoder_layers': 6},
                criterion=dict):
        super().__init__()
        
        # Model components
        self.backbone = SimpleCNNBackbone(**backbone_args)
        self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.transformer = nn.Transformer(**transformer_args)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.bbox_embed = nn.Linear(hidden_dim, 8 * 3)
        self.confidence_head = nn.Linear(hidden_dim, 2)

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self._reset_parameters()
        self.criterion = build_loss(criterion)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, rgb, **kwargs):  
        features = self.backbone(rgb)
        features = self.input_proj(features)
        B, C, H, W = features.shape
        src = features.flatten(2).permute(2, 0, 1)
        src = self.pos_encoder(src)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        # set to zeros because we do not autoregress as in NLP, but learn the queries directly
        tgt = torch.zeros_like(query_embed)
        hs = self.transformer(src, tgt)
        hs = hs.transpose(0, 1)
        pred_bboxes = self.bbox_embed(hs).view(B, -1, 8, 3)
        pred_logits = self.confidence_head(hs)
        return {'pred_bbox3d': pred_bboxes, 'pred_logits': pred_logits}
    
    def train_step(self, data, optimizer):
        self.train()
        optimizer.zero_grad()
        outputs = self(data['rgb']) # [B, num_queries, 8, 3], [B, num_queries, 2]
        losses = self.criterion(outputs, data['bbox3d'])
        losses['total_loss'].backward()
        optimizer.step()
        return losses

    def evaluate(self, data):
        self.eval()
        with torch.no_grad():
            outputs = self(data['rgb'])
            losses = self.criterion(outputs, data['bbox3d'])
        return outputs, losses