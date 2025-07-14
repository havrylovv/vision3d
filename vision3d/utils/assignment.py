import numpy as np
import torch
import torch.nn as nn
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
            gt_labels = torch.ones(
                gt_boxes.size(0), dtype=torch.int64
            )  # assume all GTs are objects

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


def test_hungarian_matcher_3d_corners():
    matcher = HungarianMatcher3D(cls_weight=1, bbox_weight=10)

    # Dummy predicted boxes: 3 predictions
    pred_bboxes = torch.tensor(
        [
            [[0, 0, 0]] * 8,  # close to GT1
            [[10, 10, 10]] * 8,  # far from any GT
            [[1, 1, 1]] * 8,  # close to GT2
        ],
        dtype=torch.float32,
    )

    # Dummy raw logits (not softmaxed)
    pred_logits = torch.tensor(
        [
            [0.1, 2.0],  # high confidence object
            [0.5, 0.5],  # uncertain
            [0.2, 1.5],  # good confidence object
        ],
        dtype=torch.float32,
    )

    # Ground truth boxes: 2 GTs
    gt_bboxes = torch.tensor(
        [
            [[0.1, 0.1, 0.1]] * 8,  # close to pred_bboxes[0]
            [[1.2, 1.1, 1.3]] * 8,  # close to pred_bboxes[2]
        ],
        dtype=torch.float32,
    )

    gt_labels = torch.tensor([1, 1], dtype=torch.int64)  # both objects

    outputs = {
        "pred_bboxes": pred_bboxes.unsqueeze(0),  # batch size 1
        "pred_logits": pred_logits.unsqueeze(0),  # batch size 1
    }

    targets = (gt_bboxes,)

    indices = matcher.match(outputs, targets)
    pred_idx, tgt_idx = indices[0]

    print("Matched prediction indices:", pred_idx.tolist())
    print("Matched ground truth indices:", tgt_idx.tolist())

    # Check that two matches found
    assert pred_idx.shape[0] == tgt_idx.shape[0] == 2, "Should match 2 predictions to 2 GT boxes"
    # Only predictions 0 and 2 should match because prediction 1 is far and uncertain
    assert set(pred_idx.tolist()) <= {0, 2}, "Only pred 0 and 2 should be matched"
    print("Test passed")


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

        return (
            self.center_weight * cost_center
            + self.size_weight * cost_size
            + self.rot_weight * cost_rot
        )

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
                indices.append(
                    (torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))
                )
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


def test_hungarian_matcher_3d_csq():
    """Test HungarianMatcher3D_CSQ with various scenarios"""

    # Initialize matcher
    matcher = HungarianMatcher3D_OBB(
        cls_weight=1.0, center_weight=2.0, size_weight=1.0, rot_weight=1.0
    )

    print("Testing HungarianMatcher3D_CSQ...")

    # Test 1: Basic matching with perfect matches
    print("\n1. Testing perfect matches...")

    # Create predictions and targets
    # Format: [cx, cy, cz, w, h, l, qx, qy, qz, qw]
    pred_boxes = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0],  # Perfect match with gt[0]
            [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],  # Perfect match with gt[1]
            [100.0, 200.0, 300.0, 0.5, 1.0, 1.5, 0.0, 0.0, 0.0, 1.0],  # No good match
        ],
        dtype=torch.float32,
    )

    gt_boxes = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0],  # Matches pred[0]
            [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],  # Matches pred[1]
        ],
        dtype=torch.float32,
    )

    # Create logits (higher for object class)
    pred_logits = torch.tensor(
        [
            [0.1, 0.9],  # High confidence object
            [0.2, 0.8],  # High confidence object
            [0.3, 0.7],  # Medium confidence object
        ],
        dtype=torch.float32,
    )

    # Prepare inputs
    outputs = {
        "pred_bbox3d": pred_boxes.unsqueeze(0),  # Add batch dimension
        "pred_logits": pred_logits.unsqueeze(0),
    }
    targets = [gt_boxes]  # List for batch

    # Run matching
    indices = matcher.match(outputs, targets)
    pred_indices, gt_indices = indices[0]

    print(f"Predicted indices: {pred_indices}")
    print(f"GT indices: {gt_indices}")

    # Check that best matches are found
    assert len(pred_indices) == 2, f"Expected 2 matches, got {len(pred_indices)}"
    assert len(gt_indices) == 2, f"Expected 2 matches, got {len(gt_indices)}"

    # Check that perfect matches are prioritized
    matches = list(zip(pred_indices.tolist(), gt_indices.tolist()))
    assert (0, 0) in matches, "Perfect match (pred 0, gt 0) not found"
    assert (1, 1) in matches, "Perfect match (pred 1, gt 1) not found"

    print("✓ Perfect matches test passed")

    # Test 2: Test with rotated boxes
    print("\n2. Testing rotated boxes...")

    # Create boxes with different rotations
    pred_boxes_rot = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # No rotation
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.707, 0.0, 0.0, 0.707],  # 90° rotation around X
        ],
        dtype=torch.float32,
    )

    gt_boxes_rot = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # No rotation - matches pred[0]
            [
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.707,
                0.0,
                0.0,
                0.707,
            ],  # 90° rotation - matches pred[1]
        ],
        dtype=torch.float32,
    )

    pred_logits_rot = torch.tensor([[0.1, 0.9], [0.1, 0.9]], dtype=torch.float32)

    outputs_rot = {
        "pred_bbox3d": pred_boxes_rot.unsqueeze(0),
        "pred_logits": pred_logits_rot.unsqueeze(0),
    }
    targets_rot = [gt_boxes_rot]

    indices_rot = matcher.match(outputs_rot, targets_rot)
    pred_indices_rot, gt_indices_rot = indices_rot[0]

    print(f"Rotation test - Predicted indices: {pred_indices_rot}")
    print(f"Rotation test - GT indices: {gt_indices_rot}")

    # Should match correctly based on rotation similarity
    assert len(pred_indices_rot) == 2, "Expected 2 rotation matches"
    print("✓ Rotation test passed")

    # Test 3: Empty targets
    print("\n3. Testing empty targets...")

    outputs_empty = {
        "pred_bbox3d": pred_boxes.unsqueeze(0),
        "pred_logits": pred_logits.unsqueeze(0),
    }
    targets_empty = [torch.empty(0, 10)]  # Empty targets

    indices_empty = matcher.match(outputs_empty, targets_empty)
    pred_indices_empty, gt_indices_empty = indices_empty[0]

    assert len(pred_indices_empty) == 0, "Should have no matches with empty targets"
    assert len(gt_indices_empty) == 0, "Should have no matches with empty targets"

    print("✓ Empty targets test passed")

    # Test 4: Multi-batch
    print("\n4. Testing multi-batch...")

    # Pad the second batch to match first batch size before stacking
    # pred_boxes: (3, 10), pred_boxes_rot: (2, 10) -> need to pad to (3, 10)
    pred_boxes_rot_padded = F.pad(pred_boxes_rot, (0, 0, 0, 1), value=0)  # Pad to (3, 10)
    pred_logits_rot_padded = F.pad(pred_logits_rot, (0, 0, 0, 1), value=0)  # Pad to (3, 2)

    # Now stack tensors of same size
    pred_boxes_batch = torch.stack([pred_boxes, pred_boxes_rot_padded])  # (2, 3, 10)
    pred_logits_batch = torch.stack([pred_logits, pred_logits_rot_padded])  # (2, 3, 2)

    outputs_batch = {"pred_bbox3d": pred_boxes_batch, "pred_logits": pred_logits_batch}
    targets_batch = [gt_boxes, gt_boxes_rot]

    indices_batch = matcher.match(outputs_batch, targets_batch)

    assert len(indices_batch) == 2, "Should return indices for both batches"
    print(f"Batch 0 matches: {len(indices_batch[0][0])}")
    print(f"Batch 1 matches: {len(indices_batch[1][0])}")

    print("✓ Multi-batch test passed")

    # Test 5: Cost component tests
    print("\n5. Testing individual cost components...")

    # Test center cost
    pred_test = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32
    )
    gt_test = torch.tensor(
        [[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32
    )

    center_cost = matcher._center_cost(pred_test, gt_test)
    print(f"Center cost (should be 1.0): {center_cost[0,0]}")
    assert abs(center_cost[0, 0] - 1.0) < 1e-6, "Center cost calculation error"

    # Test size cost
    pred_test_size = torch.tensor(
        [[0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32
    )
    gt_test_size = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32
    )

    size_cost = matcher._size_cost(pred_test_size, gt_test_size)
    print(f"Size cost (should be 1.0): {size_cost[0,0]}")
    assert abs(size_cost[0, 0] - 1.0) < 1e-6, "Size cost calculation error"

    print("✓ Cost component tests passed")

    print("\n🎉 All tests passed! HungarianMatcher3D_CSQ is working correctly.")


def test_edge_cases():
    """Test edge cases and potential issues"""

    print("\n" + "=" * 50)
    print("Testing edge cases...")

    matcher = HungarianMatcher3D_OBB()

    # Test with unnormalized quaternions
    print("\n1. Testing unnormalized quaternions...")

    pred_boxes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0],  # Unnormalized quaternion
        ],
        dtype=torch.float32,
    )

    gt_boxes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Normalized quaternion
        ],
        dtype=torch.float32,
    )

    pred_logits = torch.tensor([[0.1, 0.9]], dtype=torch.float32)

    outputs = {"pred_bbox3d": pred_boxes.unsqueeze(0), "pred_logits": pred_logits.unsqueeze(0)}
    targets = [gt_boxes]

    try:
        indices = matcher.match(outputs, targets)
        print("✓ Unnormalized quaternion test passed")
    except Exception as e:
        print(f"✗ Unnormalized quaternion test failed: {e}")

    # Test with very small numbers
    print("\n2. Testing numerical stability...")

    pred_boxes_small = torch.tensor(
        [
            [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1.0],
        ],
        dtype=torch.float32,
    )

    gt_boxes_small = torch.tensor(
        [
            [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1.0],
        ],
        dtype=torch.float32,
    )

    outputs_small = {
        "pred_bbox3d": pred_boxes_small.unsqueeze(0),
        "pred_logits": pred_logits.unsqueeze(0),
    }
    targets_small = [gt_boxes_small]

    try:
        indices_small = matcher.match(outputs_small, targets_small)
        print("✓ Numerical stability test passed")
    except Exception as e:
        print(f"✗ Numerical stability test failed: {e}")

    print("\n🎉 Edge case tests completed!")


if __name__ == "__main__":
    # test_hungarian_matcher_3d_corners()
    test_hungarian_matcher_3d_csq()
    test_edge_cases()
