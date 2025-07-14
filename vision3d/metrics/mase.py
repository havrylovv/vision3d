from typing import Any, Dict, List

import numpy as np
import torch

from vision3d.utils.registry import METRICS

from .base import Metric


@METRICS.register()
class mASEMetric(Metric):
    """
    Mean Average Scale Error metric for 3D object detection.

    Computes the scale error between predicted and ground truth
    bounding box sizes. Assumes predictions and targets are already matched.
    """

    def __init__(self):
        """Initialize mASE metric."""
        self.reset()

    def reset(self):
        """Reset the metric to its initial state."""
        self.scale_errors = []
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

        # Extract size coordinates (w, l, h) - elements 3, 4, 5
        pred_sizes = pred_bbox_np[:, 3:6]  # (N, 3)
        gt_sizes = gt_bbox_np[:, 3:6]  # (N, 3)

        # Compute scale error using 1 - min(pred/gt, gt/pred) for each dimension
        # This is the standard scale error metric used in 3D detection
        scale_errors = []
        for i in range(pred_sizes.shape[0]):
            pred_size = pred_sizes[i]  # (3,)
            gt_size = gt_sizes[i]  # (3,)

            # Avoid division by zero
            gt_size = np.maximum(gt_size, 1e-6)
            pred_size = np.maximum(pred_size, 1e-6)

            # Compute scale error for each dimension (w, l, h)
            scale_ratios = np.minimum(pred_size / gt_size, gt_size / pred_size)
            scale_error = 1 - np.mean(scale_ratios)

            scale_errors.append(scale_error)

        self.scale_errors.extend(scale_errors)
        self.num_samples += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute and return the final metric values.

        Returns:
            Dictionary containing the mASE score
        """
        if len(self.scale_errors) == 0:
            return {
                "mASE": float("inf"),
            }

        mase_score = np.mean(self.scale_errors)

        return {
            "mASE": mase_score,
        }
