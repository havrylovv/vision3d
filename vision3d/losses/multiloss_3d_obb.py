""" Multi-loss class for 3D Oriented Bounding Box (OBB) detection."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vision3d.utils.build import build_utils
from vision3d.utils.registry import LOSSES


@LOSSES.register()
class MultiLoss3D_OBB(nn.Module):
    """
    Multi-loss class for 3D Oriented Bounding Box (OBB) detection.

    Handles 10D bbox representation: [center_x, center_y, center_z, size_x, size_y, size_z, qx, qy, qz. qw]

    Combines multiple loss components:
        - Focal Loss for classification (handles class imbalance)
        - L1 Loss for center regression
        - Size Loss for dimension estimation
        - Quaternion Loss for orientation (handles rotation continuity)
        - Mask Loss for segmentation (optional)
    """

    def __init__(
        self, matcher_cfg: dict, weight_dict: Optional[dict] = None, use_mask: bool = False
    ):

        super().__init__()
        self.use_mask = use_mask
        self.matcher = build_utils(matcher_cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default weights based on common practices in 3D detection
        self.weight_dict = weight_dict or {
            "loss_center": 5.0,  # Center position loss
            "loss_size": 4.0,  # Size regression loss
            "loss_quaternion": 4.0,  # Quaternion orientation loss
            "loss_focal": 2.0,  # Focal loss for classification
            "loss_mask": 1.0,  # Mask loss for segmentation (if used)
        }

        # Focal loss parameters
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0

        # Loss functions
        self.l1_loss = nn.L1Loss(reduction="none")
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")

    def focal_loss(self, pred_logits: torch.Tensor, target_classes: torch.Tensor) -> torch.Tensor:
        """
        Focal Loss for addressing class imbalance in object detection.

        Args:
            pred_logits: (N, num_classes) prediction logits
            target_classes: (N,) target class indices

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(pred_logits, target_classes, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - p_t) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def center_loss(self, pred_center: torch.Tensor, target_center: torch.Tensor) -> torch.Tensor:
        """
        Center position loss with optional depth weighting.

        Args:
            pred_center: (N, 3) predicted centers
            target_center: (N, 3) target centers

        Returns:
            Center loss value
        """
        # Standard L1 loss for center position
        center_l1 = self.l1_loss(pred_center, target_center)

        # Optional: Weight depth (z) more heavily
        depth_weight = torch.tensor([1.0, 1.0, 1.5], device=pred_center.device)
        weighted_loss = center_l1 * depth_weight.unsqueeze(0)

        return weighted_loss.mean()

    def size_loss(self, pred_size: torch.Tensor, target_size: torch.Tensor) -> torch.Tensor:
        """
        Size loss with log-scale to handle scale variations.

        Args:
            pred_size: (N, 3) predicted sizes
            target_size: (N, 3) target sizes

        Returns:
            Size loss value
        """
        # Add small epsilon to avoid log(0)
        pred_size = torch.clamp(pred_size, min=1e-6)
        target_size = torch.clamp(target_size, min=1e-6)

        # Log-scale loss for better handling of scale variations
        log_pred = torch.log(pred_size)
        log_target = torch.log(target_size)

        return self.l1_loss(log_pred, log_target).mean()

    def quaternion_loss(self, pred_quat: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        """l
        Quaternion loss that handles rotation continuity and normalization.
        Uses geodesic distance between quaternions.

        Args:
            pred_quat: (N, 4) predicted quaternions [qx, qy, qz, qw]
            target_quat: (N, 4) target quaternions [qx, qy, qz, qw]

        Returns:
            Quaternion loss value
        """
        # Normalize quaternions
        pred_quat = F.normalize(pred_quat, p=2, dim=1)
        target_quat = F.normalize(target_quat, p=2, dim=1)

        # Compute dot product (cosine of angle between quaternions)
        dot_product = torch.sum(pred_quat * target_quat, dim=1)

        # Take absolute value to handle quaternion double cover (q and -q represent same rotation)
        dot_product = torch.abs(dot_product)

        # Clamp to avoid numerical issues
        dot_product = torch.clamp(dot_product, -1.0, 1.0)

        # Geodesic distance: 2 * arccos(|dot_product|)
        angular_distance = 2 * torch.acos(dot_product)

        return angular_distance.mean()

    def mask_loss(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """
        Mask loss for segmentation masks. Aggeregates GT into foreground/background mask.

        Args:
            pred_masks: (1, H, W) predicted masks
            target_masks: (num_objects, H, W) target masks"""

        # Convert target masks to binary foreground/background
        target_mask = torch.any(target_masks, dim=0).float()  # (H, W)
        pred_mask = pred_masks.squeeze(0)  # (H, W)

        assert (
            target_mask.shape == pred_mask.shape
        ), f"Shape mismatch: target {target_mask.shape}, pred {pred_mask.shape}"

        # Compute binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(pred_mask, target_mask, reduction="mean")

        return bce_loss

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-loss computation.

        Args:
            outputs: Dict with keys:
                - 'pred_bbox3d': (B, N, 10) predicted 3D bboxes [cx,cy,cz,sx,sy,sz,qw,qx,qy,qz]
                - 'pred_logits': (B, N, num_classes) prediction logits
                - 'pred_masks': (B, 1, H, W) predicted segmentation masks (optional) - foreground/background
            targets: List of dicts for each batch item:
                - 'bbox3d': (M, 10) target 3D bboxes
                - 'labels': (M,) target class labels
                - 'mask': List of segmentation masks (optional), each of shape [num_obj, H, W]

        Returns:
            Dictionary of loss values
        """
        # Get matching indices between predictions and targets
        indices = self.matcher.match(outputs, targets)

        # Initialize losses
        losses = {}

        # Accumulate losses across batch
        loss_center = 0.0
        loss_size = 0.0
        loss_quaternion = 0.0
        loss_focal = 0.0

        if "pred_mask" in outputs and "mask" in targets and self.use_mask:
            loss_mask = 0.0

        num_matched = 0

        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            # Extract matched predictions and targets
            pred_boxes = outputs["pred_bbox3d"][batch_idx][pred_idx]  # (num_matched, 10)
            tgt_boxes = targets["bbox3d"][batch_idx][tgt_idx]  # (num_matched, 10)

            # Center position loss
            pred_center = pred_boxes[:, :3]  # (num_matched, 3)
            tgt_center = tgt_boxes[:, :3]  # (num_matched, 3)
            loss_center += self.center_loss(pred_center, tgt_center) * len(pred_idx)

            # Size loss
            pred_size = pred_boxes[:, 3:6]  # (num_matched, 3)
            tgt_size = tgt_boxes[:, 3:6]  # (num_matched, 3)
            loss_size += self.size_loss(pred_size, tgt_size) * len(pred_idx)

            # Quaternion loss
            pred_quat = pred_boxes[:, 6:]  # (num_matched, 4)
            tgt_quat = tgt_boxes[:, 6:]  # (num_matched, 4)
            loss_quaternion += self.quaternion_loss(pred_quat, tgt_quat) * len(pred_idx)

            # Focal loss for classification
            pred_logits = outputs["pred_logits"][batch_idx]  # (num_preds, num_classes)
            target_classes = torch.zeros(
                pred_logits.size(0), dtype=torch.long, device=pred_logits.device
            )

            # Set positive class for matched predictions
            if len(tgt_idx) > 0:
                target_labels = targets["labels"][batch_idx][tgt_idx]
                target_classes[pred_idx] = target_labels

            loss_focal += self.focal_loss(pred_logits, target_classes)

            num_matched += len(pred_idx)

        # Normalize losses
        batch_size = len(targets["mask"])
        num_matched = max(num_matched, 1)  # Avoid division by zero

        # Store individual losses
        losses["loss_center"] = loss_center / num_matched
        losses["loss_size"] = loss_size / num_matched
        losses["loss_quaternion"] = loss_quaternion / num_matched
        losses["loss_focal"] = loss_focal / batch_size

        # Optional mask loss
        if "pred_mask" in outputs and "mask" in targets and self.use_mask:
            for batch_idx in range(len(targets["mask"])):
                pred_masks = outputs["pred_mask"][batch_idx]  # [1, H, W]
                tgt_masks = targets["mask"][batch_idx]  # [num_objects, H, W]
                loss_mask += self.mask_loss(pred_masks, tgt_masks)
            losses["loss_mask"] = loss_mask / batch_size

        # Compute total weighted loss
        total_loss = sum(
            self.weight_dict[k] * v for k, v in losses.items() if k in self.weight_dict
        )
        losses["total_loss"] = total_loss

        return losses
