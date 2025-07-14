"""Spatially-Aware Transformer to fuse multi-scale features from two modalities and output refined queries.

Idea: 
-> Recieves multi-scale features from two modalities (e.g. LiDAR and Camera)
-> Encodes each modality using a visual encoder with multi-scale deformable attention
-> Applies cross-attention between the two modalities
-> Applies Cross-Attention between the fused modality and learnable queries and Self-Attention 
-> Outputs refined queries for object detection

VisualEncoder part is inpired and implemented from MonoDETR: https://github.com/ZrrSkywalker/MonoDETR
"""


from typing import Optional, List, Tuple, Dict, Any
import math
import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from vision3d.models.ops.modules import MSDeformAttn
from torch.nn import MultiheadAttention 
from vision3d.utils.registry import MODELS


def generate_sine_position_embedding(pos_tensor: Tensor) -> Tensor:
    """
    Generate sinusoidal position embeddings for given position tensor.
    
    Args:
        pos_tensor: Position tensor of shape (..., pos_dim)
                   pos_dim can be 2 (x,y), 4 (x,y,w,h), or 6 (x,y,l,r,t,b)
    
    Returns:
        Sinusoidal position embeddings
    """
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    
    # Extract x, y coordinates
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    
    # Generate sine/cosine embeddings for x, y
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    
    pos_components = [pos_y, pos_x]
    
    # Handle different position dimensions
    if pos_tensor.size(-1) == 2:
        # Only x, y coordinates
        pass
    elif pos_tensor.size(-1) == 4:
        # x, y, w, h coordinates
        for i in range(2, 4):
            embed = pos_tensor[..., i] * scale
            pos_embed = embed[..., None] / dim_t
            pos_embed = torch.stack((pos_embed[..., 0::2].sin(), pos_embed[..., 1::2].cos()), dim=-1).flatten(-2)
            pos_components.append(pos_embed)
    elif pos_tensor.size(-1) == 6:
        # x, y, l, r, t, b coordinates
        for i in range(2, 6):
            embed = pos_tensor[..., i] * scale
            pos_embed = embed[..., None] / dim_t
            pos_embed = torch.stack((pos_embed[..., 0::2].sin(), pos_embed[..., 1::2].cos()), dim=-1).flatten(-2)
            pos_components.append(pos_embed)
    else:
        raise ValueError(f"Unsupported position tensor dimension: {pos_tensor.size(-1)}")
    
    return torch.cat(pos_components, dim=-1)

def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Create N identical copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def _get_activation_fn(activation: str) -> callable:
    """Return an activation function given a string identifier."""
    activation_map = {
        "relu": F.relu,
        "gelu": F.gelu,
        "glu": F.glu,
    }
    
    if activation not in activation_map:
        raise RuntimeError(f"Activation should be one of {list(activation_map.keys())}, got {activation}")
    
    return activation_map[activation]

class VisualEncoderLayer(nn.Module):
    """Single layer of the visual encoder.
        Performs multi-scale deformable attention followed by a feed-forward network.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4
    ):
        super().__init__()
        
        # Multi-scale deformable attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
    
    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        """Add positional embedding to tensor."""
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, src: Tensor) -> Tensor:
        """Forward pass through feed-forward network."""
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        return self.norm2(src)
    
    def forward(
        self,
        src: Tensor,
        pos: Optional[Tensor],
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass through encoder layer."""
        # Self-attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward network
        src = self.forward_ffn(src)
        
        return src

class VisualEncoder(nn.Module):
    """Visual encoder with multiple layers."""
    
    def __init__(self, encoder_layer: VisualEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
    
    @staticmethod
    def get_reference_points(spatial_shapes: Tensor, valid_ratios: Tensor, device: torch.device) -> Tensor:
        """Generate reference points for deformable attention."""
        reference_points_list = []
        
        for lvl, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
                indexing='ij'
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        
        return reference_points
    
    def forward(
        self,
        src: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        pos: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Forward pass through encoder.
        Args:
            src: Input feature map of shape (batch_size, channels, height, width)
            spatial_shapes: Spatial shapes of each feature map level. e.g. [(H1, W1), (H2, W2), ...]
            level_start_index: Start index for each feature map level. E.g. [0, H1*W1, H2*W2, ...]
            valid_ratios: Valid ratios for each feature map level. E.g. [(w1, h1), (w2, h2), ...]
            pos: Optional positional embeddings of shape (batch_size, channels, height, width)
            padding_mask: Optional padding mask of shape (batch_size, height, width). Used to ignore padded areas.
        Returns:
            output: Encoded feature map of shape (batch_size, channels, height, width)
        """
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        
        return output

class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention Block with Multihead Attention and Feed-Forward Network.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation: str = "relu"):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask=None, key_padding_mask=None):
        # Store original shape for later restoration
        original_query_shape = query.shape
        
        # Ensure all tensors have the shape (seq_len, batch_size, d_model)
        query = query.permute(1, 0, 2)  # (batch_size, seq_len, d_model) -> (seq_len, batch_size, d_model)
        key = key.permute(1, 0, 2)      # (batch_size, seq_len, d_model) -> (seq_len, batch_size, d_model)
        value = value.permute(1, 0, 2)  # (batch_size, seq_len, d_model) -> (seq_len, batch_size, d_model)

        # Perform cross-attention
        attn_output, _ = self.cross_attn(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        query = query + self.dropout(attn_output)
        query = self.norm1(query)

        # Feed-forward network
        ffn_output = self.ffn(query)
        query = query + self.dropout(ffn_output)
        query = self.norm2(query)

        # Restore original shape: (seq_len, batch_size, d_model) -> (batch_size, seq_len, d_model)
        query = query.permute(1, 0, 2)
        
        return query

class DecoderLayer(nn.Module):
    """
    Decoder layer with self-attention and cross-attention.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation: str = "relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_block = CrossAttentionBlock(d_model, nhead, dim_feedforward, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries: torch.Tensor, memory: torch.Tensor, attn_mask=None, key_padding_mask=None):
        """
        Args:
            queries: (batch_size, num_queries, d_model)
            memory: (batch_size, seq_len, d_model)
        """
        # Self-attention among queries
        q_permuted = queries.permute(1, 0, 2)  # (num_queries, batch_size, d_model)
        self_attn_out, _ = self.self_attn(q_permuted, q_permuted, q_permuted, attn_mask=attn_mask)
        queries = queries + self.dropout(self_attn_out.permute(1, 0, 2))  # Back to (batch_size, num_queries, d_model)
        queries = self.norm1(queries)
        
        # Cross-attention with memory
        queries = self.cross_attn_block(queries, memory, memory)
        
        return queries
    
@MODELS.register()
class SpatiallyAwareTransformer(nn.Module):
    """
    Spatially-Aware Transformer for object detection.
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        num_feature_levels: int = 4,
        enc_n_points: int = 4,
        cross_attn_layers: int = 3,
        query_dim: int = 256,
        num_queries: int = 100,
        memory_pool_dim: int = 512,
        decoder_layers: int = 6
    ):
        """
        Initialize Spatially-Aware Transformer.
        
        Args:
            d_model: Hidden dimension.
            nhead: Number of attention heads.
            num_encoder_layers: Number of encoder layers for each modality.
            dim_feedforward: Feed-forward network dimension.
            dropout: Dropout rate.
            activation: Activation function ('relu', 'gelu', 'glu').
            num_feature_levels: Number of feature pyramid levels.
            enc_n_points: Number of sampling points in encoder.
            cross_attn_layers: Number of cross-attention layers between modalities.
            query_dim: Dimension of learnable queries.
            num_queries: Number of learnable queries.
            memory_pool_dim: Dimension of the memory pool after adaptive pooling.
            decoder_layers: Number of decoder layers for progressive refinement.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.memory_pool_dim = memory_pool_dim

        # Encoders for two modalities
        self.encoder1 = self._build_encoder(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points, num_encoder_layers
        )
        self.encoder2 = self._build_encoder(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points, num_encoder_layers
        )

        # Adaptive pooling to reduce memory dimension - is not supported by ONNX, thus, replaced with interpolation 
        #self.memory_pool = nn.AdaptiveAvgPool1d(self.memory_pool_dim)
        self.memory_pool = F.interpolate
        
        # Learnable weight for fusing features from two modalities
        self.alpha = nn.Parameter(torch.tensor(0.5, device=self.device))  


        # Cross-attention layers between modalities
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dim_feedforward, dropout, activation) for _ in range(cross_attn_layers)
        ])

        # Decoder layers for progressive refinement
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout, activation) for _ in range(decoder_layers)
        ])

        # Cross-attention between fused modality and learnable queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, query_dim))
        
        # Level embeddings for multi-scale features
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        
        # Initialize parameters
        self._reset_parameters()

    def _build_encoder(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        num_feature_levels: int,
        nhead: int,
        enc_n_points: int,
        num_encoder_layers: int,
    ) -> VisualEncoder:
        """Helper function to build an encoder."""
        encoder_layer = VisualEncoderLayer(
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=enc_n_points
        )
        return VisualEncoder(encoder_layer, num_encoder_layers)

    def _reset_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        normal_(self.level_embed)


    def get_valid_ratio(self, mask: Tensor) -> Tensor:
        """Calculate valid ratio for each feature map."""
        _, height, width = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / height
        valid_ratio_w = valid_W.float() / width
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    def prepare_inputs(
        self,
        srcs: List[Tensor],
        masks: List[Tensor],
        pos_embeds: List[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Prepare inputs for the encoder.
        """
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            batch_size, channels, height, width = src.shape
            spatial_shapes.append((height, width))
            
            # Flatten spatial dimensions
            src = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            
            # Add level embedding
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            
            # Flatten mask
            mask = mask.flatten(1)
            
            # Collect flattened inputs
            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
        
        # Concatenate all levels
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs[0].device)
        
        # Compute level start index
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ))
        
        return src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index

    def forward(
        self,
        inputs1: Dict[str, List[Tensor]],
        inputs2: Dict[str, List[Tensor]],
    ) -> Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            inputs1: Dictionary containing `srcs`, `masks`, and `pos_embeds` for the first encoder.
            inputs2: Dictionary containing `srcs`, `masks`, and `pos_embeds` for the second encoder.
        
        Returns:
            Final output after cross-attention with learnable queries.
        """
        # Prepare inputs for encoder 1
        src_flatten1, mask_flatten1, lvl_pos_embed_flatten1, spatial_shapes1, level_start_index1 = self.prepare_inputs(
            inputs1["srcs"], inputs1["masks"], inputs1["pos_embeds"]
        )
        valid_ratios1 = torch.stack([self.get_valid_ratio(m) for m in inputs1["masks"]], 1)
        
        # Prepare inputs for encoder 2
        src_flatten2, mask_flatten2, lvl_pos_embed_flatten2, spatial_shapes2, level_start_index2 = self.prepare_inputs(
            inputs2["srcs"], inputs2["masks"], inputs2["pos_embeds"]
        )
        valid_ratios2 = torch.stack([self.get_valid_ratio(m) for m in inputs2["masks"]], 1)
        
        # Forward pass through both encoders
        memory1 = self.encoder1(
            src_flatten1, spatial_shapes1, level_start_index1, valid_ratios1,
            lvl_pos_embed_flatten1, mask_flatten1
        )
        memory2 = self.encoder2(
            src_flatten2, spatial_shapes2, level_start_index2, valid_ratios2,
            lvl_pos_embed_flatten2, mask_flatten2
        )

        # Apply adaptive pooling to reduce memory dimension - is not supported by ONNX, thus, replaced with interpolation
        #memory1 = self.memory_pool(memory1.permute(0, 2, 1)).permute(0, 2, 1)
        #memory2 = self.memory_pool(memory2.permute(0, 2, 1 )).permute(0, 2, 1)
        memory1 = self.memory_pool(memory1.permute(0, 2, 1), size=self.memory_pool_dim, mode='linear', align_corners=False).permute(0, 2, 1)
        memory2 = self.memory_pool(memory2.permute(0, 2, 1), size=self.memory_pool_dim, mode='linear', align_corners=False).permute(0, 2, 1)


        # Apply cross-attention blocks
        for cross_attn_block in self.cross_attn_blocks:
            memory1 = cross_attn_block(memory1, memory2, memory2)
            memory2 = cross_attn_block(memory2, memory1, memory1)   

        fused_memory = self.alpha * memory1 + (1 - self.alpha) * memory2

        # Convert query_embed to the expected input format for DecoderLayer
        queries = self.query_embed.unsqueeze(1).repeat(1, fused_memory.size(0), 1)  # (num_queries, batch_size, query_dim)
        queries = queries.permute(1, 0, 2)  # (batch_size, num_queries, query_dim)
        
        # Apply multiple decoder layers for progressive refinement
        for decoder_layer in self.decoder_layers:
            queries = decoder_layer(queries, fused_memory)

        return queries

def test_transformer():
    """
    Test the transformer with dummy inputs.
    """
    batch_size = 2
    d_model = 256
    seq_len = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2
    d_model = 256
    feature_map_shapes = [(32, 32), (16, 16), (8, 8), (4, 4)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer = SpatiallyAwareTransformer(
        d_model=d_model,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        activation="relu",
        cross_attn_layers=6,
        query_dim=d_model,
        num_queries=50,
        memory_pool_dim=512
    ).to(device)

    # masks: 0 - keep, 1 - ignore
    inputs1 = {
        "srcs": [torch.randn(batch_size, d_model, h, w).to(device) for h, w in feature_map_shapes],
        "masks": [torch.zeros(batch_size, h, w, dtype=torch.bool).to(device).bool() for h, w in feature_map_shapes],
        "pos_embeds": [torch.randn(batch_size, d_model, h, w).to(device) for h, w in feature_map_shapes],
    }
    inputs2 = {
        "srcs": [torch.randn(batch_size, d_model, h, w).to(device) for h, w in feature_map_shapes],
        "masks": [torch.zeros(batch_size, h, w, dtype=torch.bool).to(device).bool() for h, w in feature_map_shapes],
        "pos_embeds": [torch.randn(batch_size, d_model, h, w).to(device) for h, w in feature_map_shapes],
    }

    print("Running transformer test...")
    final_output = transformer(inputs1, inputs2)
    print("Transformer test completed.")
    print(f"Final output shape: {final_output.shape}")

if __name__ == "__main__":
    test_transformer()