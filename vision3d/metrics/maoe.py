"""mAOE Metric for 3D Object Detection"""

from typing import Any, Dict, List

import numpy as np
import torch

from vision3d.utils.registry import METRICS

from .base import Metric


@METRICS.register()
class mAOEMetric(Metric):
    """
    Mean Average Orientation Error metric for 3D object detection.

    Computes the orientation error between predicted and ground truth
    bounding box orientations. Assumes predictions and targets are already matched.
    """

    def __init__(self):
        """Initialize mAOE metric."""
        self.reset()

    def reset(self):
        """Reset the metric to its initial state."""
        self.orientation_errors = []
        self.num_samples = 0

    def update(self, preds: Dict[str, List[torch.Tensor]], targets: Dict[str, List[torch.Tensor]]):
        """
        Update the metric with predictions and targets.

        Args:
            preds: Dictionary containing:
                   'pred_bbox3d': List of torch tensors, each of shape (N, 10)
                                 - [cx, cy, cz, w, l, h, qx, qy, qz, qw]
                   'pred_logits': List of torch tensors, each of shape (N, num_classes)
            targets: Dictionary containing:
                    'bbox3d': List of torch tensors, each of shape (N, 10)
                             - [cx, cy, cz, w, l, h, qx, qy, qz, qw]
                    'labels': List of torch tensors, each of shape (N, 1)
        """
        pred_bboxes = preds["pred_bbox3d"]  # List of (N, 10)

        gt_bboxes = targets["bbox3d"]  # List of (N, 10)

        assert len(pred_bboxes) == len(gt_bboxes), "Predictions and targets must have the same number of samples."

        for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
            self._update_single_sample(pred_bbox, gt_bbox)

    def _update_single_sample(self, pred_bbox: torch.Tensor, gt_bbox: torch.Tensor):
        """
        Update metric for a single sample.

        Args:
            pred_bbox: Predicted bounding boxes (N, 10)
            gt_bbox: Ground truth bounding boxes (N, 10)
        """
        if pred_bbox.shape[0] == 0 or gt_bbox.shape[0] == 0:
            self.num_samples += 1
            return

        pred_bbox_np = pred_bbox.detach().cpu().numpy()
        gt_bbox_np = gt_bbox.detach().cpu().numpy()

        # Extract quaternions (qx, qy, qz, qw) - elements 6, 7, 8, 9
        pred_quats = pred_bbox_np[:, 6:10]  # (N, 4)
        gt_quats = gt_bbox_np[:, 6:10]  # (N, 4)

        # Compute orientation error using quaternion angle difference
        orientation_errors = []
        for i in range(pred_quats.shape[0]):
            pred_quat = pred_quats[i]  # (4,) - [qx, qy, qz, qw]
            gt_quat = gt_quats[i]  # (4,) - [qx, qy, qz, qw]

            # Normalize quaternions
            pred_quat = pred_quat / (np.linalg.norm(pred_quat) + 1e-6)
            gt_quat = gt_quat / (np.linalg.norm(gt_quat) + 1e-6)

            # Compute the angle between quaternions
            # Using the formula: angle = 2 * arccos(|q1 · q2|)
            dot_product = np.abs(np.dot(pred_quat, gt_quat))
            dot_product = np.clip(dot_product, 0.0, 1.0)  # Clamp to avoid numerical errors

            angle_error = 2 * np.arccos(dot_product)

            # Ensure angle is in [0, π] range
            angle_error = min(angle_error, np.pi - angle_error)

            orientation_errors.append(angle_error)

        self.orientation_errors.extend(orientation_errors)
        self.num_samples += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute and return the final metric values.

        Returns:
            Dictionary containing the mAOE score
        """
        if len(self.orientation_errors) == 0:
            return {
                "mAOE": float("inf"),
            }

        maoe_score = np.mean(self.orientation_errors)

        return {
            "mAOE": maoe_score,
        }
