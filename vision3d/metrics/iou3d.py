"""IoU3D Metric Implementation"""

from typing import Any, Dict, List

import numpy as np
import torch

from vision3d.utils.bbox_converter import obb_to_corners
from vision3d.utils.registry import METRICS

from .base import Metric
from .ops.iou3d import IoU3D as IoU3D_Ops


@METRICS.register()
class IoU3DMetric(Metric):
    """
    3D Intersection over Union metric for 3D object detection.

    Computes the 3D IoU between predicted and ground truth
    bounding boxes. Assumes predictions and targets are already matched.
    """

    def __init__(self):
        """Initialize IoU3D metric."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iou3d = IoU3D_Ops(self.device)
        self.reset()

    def reset(self):
        """Reset the metric to its initial state."""
        self.iou_values = []
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
            gt_bbox: Ground truth bounding boxes (N, 10)
        """
        if pred_bbox.shape[0] == 0 or gt_bbox.shape[0] == 0:
            self.num_samples += 1
            return

        # Convert bboxes to corner format for IoU computation
        pred_corners = obb_to_corners(pred_bbox)  # (N, 8, 3)
        gt_corners = obb_to_corners(gt_bbox)  # (N, 8, 3)

        # Compute 3D IoU
        iou_values = self.iou3d.compute_iou_3d(pred_corners, gt_corners)  # (N,)

        self.iou_values.extend(iou_values.tolist())
        self.num_samples += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute and return the final metric values.

        Returns:
            Dictionary containing the IoU3D score
        """
        if len(self.iou_values) == 0:
            return {
                "IoU3D": 0.0,
            }

        iou3d_score = np.mean(self.iou_values)

        return {
            "mIoU3D": iou3d_score,
        }
