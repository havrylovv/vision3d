from typing import Any, Dict, List

import numpy as np
import torch

from vision3d.utils.registry import METRICS

from .base import Metric


@METRICS.register()
class mATEMetric(Metric):
    """
    Mean Average Translation Error metric for 3D object detection.

    Computes the euclidean distance error between predicted and ground truth
    bounding box centers. Assumes predictions and targets are already matched.
    """

    def __init__(self):
        """Initialize mATE metric."""
        self.reset()

    def reset(self):
        """Reset the metric to its initial state."""
        self.translation_errors = []
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

        assert len(pred_bboxes) == len(
            gt_bboxes
        ), "Predictions and targets must have the same number of samples."

        for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
            self._update_single_sample(pred_bbox, gt_bbox)

    def _update_single_sample(self, pred_bbox: torch.Tensor, gt_bbox: torch.Tensor):
        """
        Update metric for a single sample.

        Args:
            pred_bbox: Predicted bounding boxes (N, 10)
            pred_logit: Predicted logits (N, num_classes)
        """
        if pred_bbox.shape[0] == 0 or gt_bbox.shape[0] == 0:
            self.num_samples += 1
            return

        pred_bbox_np = pred_bbox.detach().cpu().numpy()
        gt_bbox_np = gt_bbox.detach().cpu().numpy()

        # Extract center coordinates (cx, cy, cz) - first 3 elements
        pred_centers = pred_bbox_np[:, :3]  # (N, 3)
        gt_centers = gt_bbox_np[:, :3]  # (N, 3)

        translation_errors = np.sqrt(np.sum((pred_centers - gt_centers) ** 2, axis=1))
        self.translation_errors.extend(translation_errors.tolist())

        self.num_samples += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute and return the final metric values.

        Returns:
            Dictionary containing the mATE score
        """
        if len(self.translation_errors) == 0:
            return {
                "mATE": float("inf"),
            }

        mate_score = np.mean(self.translation_errors)

        return {
            "mATE": mate_score,
        }
