"""Core evaluator class for computing evaluation metrics in 3D vision tasks."""

from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader

from vision3d.models.base import Vision3DModel
from vision3d.utils.build import build_metric, build_utils
from vision3d.utils.misc import to_device
from vision3d.utils.registry import UTILS


@UTILS.register()
class Evaluator:
    """
    Evaluator class for computing evaluation metrics for 3D vision tasks.

    Can be used standalone or integrated into the trainer for validation.
    """

    def __init__(
        self, matcher: dict, metric_cfgs: List[dict], device: Union[str, torch.device] = "cpu"
    ):
        """
        Initialize the evaluator.

        Args:
            matcher: Matcher configuration dictionary
            metric_cfgs: List of metric configuration dictionaries
            device: Device to run evaluation on
        """
        self.matcher = matcher
        self.matcher = build_utils(matcher)
        self.metrics = self._build_metrics(metric_cfgs)
        self.device = device

    def _build_metrics(self, metric_cfgs: List[dict]) -> List:
        """Build metrics from configuration."""
        metrics = []
        for cfg in metric_cfgs:
            metric = build_metric(cfg)
            metrics.append(metric)
        return metrics

    def reset(self):
        """Reset all metrics to initial state."""
        for metric in self.metrics:
            metric.reset()

    def update(self, outputs: Dict[str, Any], targets: Dict[str, Any]):
        """
        Update metrics with a batch of predictions and targets.

        Args:
            outputs: Model outputs containing 'pred_bbox3d' and 'pred_logits' keys
            targets: Ground truth targets containing 'bbox3d' and 'labels' keys
        """
        # Get matched indices from matcher
        indices = self.matcher.match(outputs, targets)

        # Prepare matched data for metrics
        matched_preds = {"pred_bbox3d": [], "pred_logits": []}
        matched_targets = {"bbox3d": [], "labels": []}

        # Extract matched predictions and targets for each batch
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                # No matches for this batch - add empty tensors
                matched_preds["pred_bbox3d"].append(torch.empty(0, 10, device=self.device))
                matched_preds["pred_logits"].append(
                    torch.empty(0, outputs["pred_logits"][batch_idx].shape[-1], device=self.device)
                )
                matched_targets["bbox3d"].append(torch.empty(0, 10, device=self.device))
                matched_targets["labels"].append(torch.empty(0, 1, device=self.device))
                continue

            # Extract matched predictions and targets
            pred_boxes = outputs["pred_bbox3d"][batch_idx][pred_idx]  # (num_matched, 10)
            pred_logits = outputs["pred_logits"][batch_idx][pred_idx]  # (num_matched, num_classes)
            tgt_boxes = targets["bbox3d"][batch_idx][tgt_idx]  # (num_matched, 10)
            tgt_labels = targets["labels"][batch_idx][tgt_idx]  # (num_matched,)

            # Ensure labels have correct shape
            if tgt_labels.dim() == 1:
                tgt_labels = tgt_labels.unsqueeze(-1)  # (num_matched, 1)

            matched_preds["pred_bbox3d"].append(pred_boxes)
            matched_preds["pred_logits"].append(pred_logits)
            matched_targets["bbox3d"].append(tgt_boxes)
            matched_targets["labels"].append(tgt_labels)

        # Update all metrics
        for metric in self.metrics:
            metric.update(matched_preds, matched_targets)

    def compute(self) -> Dict[str, Any]:
        """
        Compute final metric values.

        Returns:
            Dictionary mapping metric names to their computed values
        """
        results = {}
        for metric in self.metrics:
            metric_results = metric.compute()
            # Check if metric has a custom name, otherwise use class name
            if hasattr(metric, "name") and metric.name:
                metric_name = metric.name
            else:
                metric_name = metric.__class__.__name__

            # Handle different types of metric results
            if isinstance(metric_results, dict):
                # If metric returns a dict, add each key-value pair directly
                for key, value in metric_results.items():
                    results[key] = value
            else:
                # If metric returns a single value, use metric name as key
                results[metric_name] = metric_results
        return results

    def evaluate_dataloader(
        self, model: Vision3DModel, dataloader: DataLoader, reset_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a complete dataloader.

        Args:
            model: Model to evaluate
            dataloader: DataLoader to evaluate on
            reset_metrics: Whether to reset metrics before evaluation

        Returns:
            Dictionary of computed metrics
        """
        if reset_metrics:
            self.reset()

        model.eval()
        with torch.no_grad():
            for inputs, targets in dataloader:
                # Move to device
                inputs = {k: to_device(v, self.device) for k, v in inputs.items()}
                targets = {k: to_device(v, self.device) for k, v in targets.items()}

                # Get model outputs
                outputs, _ = model.evaluate(inputs, targets)

                # Update metrics
                self.update(outputs, targets)

        return self.compute()

    def evaluate_single_batch(
        self, model: Vision3DModel, inputs: Dict[str, Any], targets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a single batch.

        Args:
            model: Model to evaluate
            inputs: Input batch
            targets: Target batch

        Returns:
            Dictionary of computed metrics for this batch
        """
        model.eval()
        with torch.no_grad():
            # Move to device
            inputs = {k: to_device(v, self.device) for k, v in inputs.items()}
            targets = {k: to_device(v, self.device) for k, v in targets.items()}

            # Get model outputs
            outputs, _ = model.evaluate(inputs, targets)

            # Create temporary evaluator for single batch
            temp_evaluator = Evaluator(self.matcher, [], self.device)
            temp_evaluator.metrics = [type(metric)() for metric in self.metrics]

            # Update and compute
            temp_evaluator.update(outputs, targets)
            return temp_evaluator.compute()
