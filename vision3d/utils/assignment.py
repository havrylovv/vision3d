"""Scipy-based Hungarian matcher for 3D bounding boxes represented as corners or oriented bounding boxes (OBB)."""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from vision3d.utils.registry import UTILS


@UTILS.register()
class HungarianMatcher3D_Corners:
    def __init__(self, cls_weight=1, bbox_weight=5):
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight

    def _bbox_cost(self, pred_boxes: Tensor, gt_boxes: Tensor):
        pred_flat = pred_boxes.view(pred_boxes.shape[0], -1)  # (num_preds, 24)
        gt_flat = gt_boxes.view(gt_boxes.shape[0], -1)  # (num_gt, 24)
        cost = torch.cdist(pred_flat, gt_flat, p=1)  # (num_preds, num_gt)
        return cost.detach().cpu().numpy()

    def _cls_cost(self, pred_logits: Tensor, gt_labels: Tensor):
        # pred_logits: (num_preds, 2), gt_labels: (num_gt,)
        pred_probs = torch.softmax(pred_logits, dim=-1)  # (num_preds, 2)
        object_prob = pred_probs[:, 1]  # confidence for "object" class
        cost = -object_prob.unsqueeze(1).expand(-1, gt_labels.size(0))
        return cost.detach().cpu().numpy()

    def match(self, outputs, targets):
        indices = []
        for batch_idx in range(len(targets)):
            pred_boxes = outputs["pred_bbox3d"][batch_idx]  # (num_preds, 8, 3)
            pred_logits = outputs["pred_logits"][batch_idx]  # (num_preds, 2)

            gt_boxes = targets[batch_idx]  # (num_gt, 8, 3)
            gt_labels = torch.ones(gt_boxes.size(0), dtype=torch.int64)  # assume all GTs are objects

            cost_bbox = self._bbox_cost(pred_boxes, gt_boxes)
            cost_cls = self._cls_cost(pred_logits, gt_labels)

            cost_matrix = self.cls_weight * cost_cls + self.bbox_weight * cost_bbox
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64),
                    torch.as_tensor(col_ind, dtype=torch.int64),
                )
            )
        return indices


@UTILS.register()
class HungarianMatcher3D_OBB:
    """
    Hungarian matcher for 3D bounding boxes in [center, size, quaternion] format.
    Box format: [cx, cy, cz, w, h, l, qx, qy, qz, qw] (10 dimensions)
    """

    def __init__(self, cls_weight=1, center_weight=2, size_weight=1, rot_weight=1):
        self.cls_weight = cls_weight
        self.center_weight = center_weight
        self.size_weight = size_weight
        self.rot_weight = rot_weight

    def _center_cost(self, pred_boxes: Tensor, gt_boxes: Tensor):
        """L1 distance between centers"""
        pred_centers = pred_boxes[:, :3]  # (num_preds, 3)
        gt_centers = gt_boxes[:, :3]  # (num_gt, 3)
        cost = torch.cdist(pred_centers, gt_centers, p=1)  # (num_preds, num_gt)
        return cost.detach().cpu().numpy()

    def _size_cost(self, pred_boxes: Tensor, gt_boxes: Tensor):
        """L1 distance between sizes"""
        pred_sizes = pred_boxes[:, 3:6]  # (num_preds, 3)
        gt_sizes = gt_boxes[:, 3:6]  # (num_gt, 3)

        cost = torch.cdist(pred_sizes, gt_sizes, p=1)  # (num_preds, num_gt)
        return cost.detach().cpu().numpy()

    def _rotation_cost(self, pred_boxes: Tensor, gt_boxes: Tensor):
        """Angular distance between quaternions"""
        pred_quats = pred_boxes[:, 6:10]  # (num_preds, 4)
        gt_quats = gt_boxes[:, 6:10]  # (num_gt, 4)

        # Normalize quaternions
        pred_quats = F.normalize(pred_quats, p=2, dim=1)
        gt_quats = F.normalize(gt_quats, p=2, dim=1)

        # Compute angular distance using dot product
        # |q1 · q2| gives cosine of half the angle between rotations
        dot_product = torch.abs(torch.mm(pred_quats, gt_quats.t()))  # (num_preds, num_gt)

        # Convert to angular distance: arccos(|q1 · q2|) * 2
        # Clamp to avoid numerical issues
        dot_product = torch.clamp(dot_product, 0, 1)
        angular_dist = torch.acos(dot_product) * 2  # (num_preds, num_gt)

        return angular_dist.detach().cpu().numpy()

    def _bbox_cost_combined(self, pred_boxes: Tensor, gt_boxes: Tensor):
        """Combined cost using all bbox components"""
        cost_center = self._center_cost(pred_boxes, gt_boxes)
        cost_size = self._size_cost(pred_boxes, gt_boxes)
        cost_rot = self._rotation_cost(pred_boxes, gt_boxes)

        return self.center_weight * cost_center + self.size_weight * cost_size + self.rot_weight * cost_rot

    def _cls_cost(self, pred_logits: Tensor, gt_labels: Tensor):
        """Classification cost"""
        pred_probs = torch.softmax(pred_logits, dim=-1)  # (num_preds, num_classes)

        # If binary classification (background vs object)
        if pred_probs.shape[-1] == 2:
            object_prob = pred_probs[:, 1]  # confidence for "object" class
            cost = -object_prob.unsqueeze(1).expand(-1, gt_labels.size(0))
        else:
            # Multi-class case
            cost = -pred_probs[:, gt_labels].t()  # (num_preds, num_gt)

        return cost.detach().cpu().numpy()

    def match(self, outputs: dict, targets: dict):
        """
        Args:
            outputs: Dict with 'pred_bbox3d' (B, N, 10) and 'pred_logits' (B, N, num_classes)
            targets: Dict with 'bbox3d' (B, num_gt, 10) for each batch and 'labels' (B,) of index of object class.

        Returns:
            List of (pred_indices, gt_indices) tuples for each batch
        """
        indices = []

        for batch_idx in range(len(targets["bbox3d"])):
            pred_boxes = outputs["pred_bbox3d"][batch_idx]  # (num_preds, 10)
            pred_logits = outputs["pred_logits"][batch_idx]  # (num_preds, 1)

            gt_boxes = targets["bbox3d"][batch_idx]  # (num_gt, 10)

            # Skip if no ground truth boxes
            if gt_boxes.size(0) == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            # Assume all GTs are objects (or extract from targets if available)
            gt_labels = targets["labels"][batch_idx]

            # Compute costs
            cost_bbox = self._bbox_cost_combined(pred_boxes, gt_boxes)
            cost_cls = self._cls_cost(pred_logits, gt_labels)

            # Total cost matrix
            cost_matrix = self.cls_weight * cost_cls + cost_bbox

            # Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64),
                    torch.as_tensor(col_ind, dtype=torch.int64),
                )
            )

        return indices
