import torch
from torch import Tensor 
from torch import nn
from vision3d.models.base import Vision3DModel
from typing import Optional

from vision3d.utils.build import build_model, build_loss
from typing import List, Dict, Tuple
from vision3d.utils.registry import MODELS
from vision3d.models.modelling.utils.pos_embed import PositionEmbeddingSine
    
@MODELS.register()
class MonoDETR3D(Vision3DModel):
    def __init__(self,
                image_encoder: dict,
                pc_encoder: dict,
                fusion_transformer: dict, 
                bbox_head: dict,
                criterion: dict,
                mask_head: Optional[dict] = None,
    ): 
        super(MonoDETR3D, self).__init__()

        self.image_encoder = build_model(image_encoder)
        self.pc_encoder = build_model(pc_encoder)
        self.fusion_transformer = build_model(fusion_transformer)
        self.bbox_head = build_model(bbox_head)
        self.criterion = build_loss(criterion)
        self.mask_head = build_model(mask_head) if mask_head is not None else None

        # position embeddings
        hidden_dim = getattr(self.image_encoder, 'output_dim', None)       
        if hidden_dim is None:
            raise ValueError("output_dim must be specified in image_encoder config")
        N_steps = hidden_dim // 2
        self.image_position_embedding = PositionEmbeddingSine(N_steps, normalize=True)  
        self.pc_position_embedding = PositionEmbeddingSine(N_steps, normalize=True) 


    def forward(self, image: Tensor, point_cloud: Tensor) -> dict:
        """
        Forward pass for MonoDETR3D model.
        
        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W).
            point_cloud (Tensor): Input point cloud tensor of shape (B, 3, H, W).
        
        Returns:
            dict: Dictionary containing the outputs from bbox head and mask head (if applicable).
        """

        # Encode image and point cloud: (multi scale features)
        image_features = self.image_encoder(image)
        pc_features = self.pc_encoder(point_cloud)

        # Prepare Multi-scale features (features, masks, positional embeddings)
        image_inputs, pc_inputs = self.prepare_inputs_to_fusion_transformer(
            point_cloud, image_features, pc_features
        )

        #import pdb; pdb.set_trace()  # Debugging breakpoint
        # Fuse features and get queries     
        fused_features = self.fusion_transformer(image_inputs, pc_inputs)

        # Bbox head outputs
        outputs = self.bbox_head(fused_features)

        if self.mask_head is not None:
            # Predict from image encoder features directly 
            mask_outputs = self.mask_head(image_features)
            outputs.update(mask_outputs)
            
        return outputs

    def prepare_inputs_to_fusion_transformer(self,
                                            point_cloud: Tensor, 
                                            image_features: List[Tensor],
                                            pc_features: List[Tensor]) -> Tuple[Dict[str, List[Tensor]], Dict[str, List[Tensor]]]:
        """Aggregate multi-scale features, generates masks for Defformable Self-Attention, and positional embeddings.
        For images, we attend to all pixels, while for point clouds, we generate masks to ignore missing points.
        """
        B = point_cloud.shape[0]

        assert isinstance(image_features, dict), "Image features should be a dictionary of multi-scale features."
        assert isinstance(pc_features, dict), "Point cloud features should be a dictionary of multi-scale features."
        
        image_features = [single_scale for single_scale in image_features.values()]
        pc_features = [single_scale for single_scale in pc_features.values()]
        
        features_shapes = [feat.shape[-2:] for feat in image_features]  # [(H1, W1), (H2, W2), ...]

        # Generate masks not to attend to missing points in point cloud (0 - keep, 1 - remove)
        missing_mask = ~torch.all(point_cloud == 0, dim=1, keepdim=True)  # (B, 1, H, W)
        pc_masks = [
            nn.functional.interpolate(missing_mask.float(), size=shape, mode='bilinear', align_corners=False).bool().squeeze(1)
            for shape in features_shapes
        ]
        # Generate masks for image features (attend to all pixels)
        image_masks = [torch.zeros((B, *feat_shape), dtype=torch.bool, device=missing_mask.device) for feat_shape in features_shapes]

         # Generate positional embeddings for both modalities
        image_pos_embeds = self.generate_position_embeddings(
            image_features, image_masks, self.image_position_embedding
        )
        pc_pos_embeds = self.generate_position_embeddings(
            pc_features, pc_masks, self.pc_position_embedding
        )
        
        # Prepare inputs for fusion transformer
        image_inputs = {
            "srcs": image_features,
            "masks": image_masks,
            "pos_embeds": image_pos_embeds
        }
        
        pc_inputs = {
            "srcs": pc_features,
            "masks": pc_masks,
            "pos_embeds": pc_pos_embeds
        }

        return image_inputs, pc_inputs
    

    def generate_position_embeddings(self, features: List[Tensor], masks: List[Tensor], 
                                   position_embedding_module) -> List[Tensor]:
        """
        Generate positional embeddings for multi-scale features.
        
        Args:
            features: List of feature tensors at different scales
            masks: List of corresponding masks
            position_embedding_module: Position embedding module to use
            
        Returns:
            List of positional embeddings for each scale
        """
        pos_embeds = []
        
        for feat, mask in zip(features, masks):
            # Generate position embedding directly with feature and mask
            pos_embed = position_embedding_module(feat, mask)
            pos_embeds.append(pos_embed)
            
        return pos_embeds
    
    def train_step(self, inputs, targets, optimizer):
        self.train()
        optimizer.zero_grad()
        outputs = self(inputs['rgb'], inputs['pc'])                       
        losses = self.criterion(outputs, targets)
        losses['total_loss'].backward()
        optimizer.step()
        return losses

    def evaluate(self, inputs, targets):
        self.eval()
        with torch.no_grad():
            outputs = self(inputs['rgb'], inputs['pc'])
            losses = self.criterion(outputs, targets)
        return outputs, losses  