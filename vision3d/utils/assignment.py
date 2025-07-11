import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from scipy.optimize import linear_sum_assignment
from vision3d.utils.registry import UTILS

@UTILS.register()
class HungarianMatcher3D:
    def __init__(self, cls_weight=1, bbox_weight=5):
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight

    def _bbox_cost(self, pred_boxes: Tensor, gt_boxes: Tensor):
        pred_flat = pred_boxes.view(pred_boxes.shape[0], -1)  # (num_preds, 24)
        gt_flat = gt_boxes.view(gt_boxes.shape[0], -1)        # (num_gt, 24)
        cost = torch.cdist(pred_flat, gt_flat, p=1)           # (num_preds, num_gt)
        return cost.detach().cpu().numpy()

    def _cls_cost(self, pred_logits: Tensor, gt_labels: Tensor):
        # pred_logits: (num_preds, 2), gt_labels: (num_gt,)
        pred_probs = torch.softmax(pred_logits, dim=-1)       # (num_preds, 2)
        object_prob = pred_probs[:, 1]                        # confidence for "object" class
        cost = -object_prob.unsqueeze(1).expand(-1, gt_labels.size(0))
        return cost.detach().cpu().numpy()

    def match(self, outputs, targets):
        indices = []
        for batch_idx in range(len(targets)):
            pred_boxes = outputs['pred_bbox3d'][batch_idx]               # (num_preds, 8, 3)
            pred_logits = outputs['pred_logits'][batch_idx]              # (num_preds, 2)

            gt_boxes = targets[batch_idx]                                # (num_gt, 8, 3)
            gt_labels = torch.ones(gt_boxes.size(0), dtype=torch.int64)  # assume all GTs are objects
            
            cost_bbox = self._bbox_cost(pred_boxes, gt_boxes)
            cost_cls = self._cls_cost(pred_logits, gt_labels)

            cost_matrix = self.cls_weight * cost_cls + self.bbox_weight * cost_bbox
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            indices.append((torch.as_tensor(row_ind, dtype=torch.int64),
                            torch.as_tensor(col_ind, dtype=torch.int64)))
        return indices



def test_hungarian_matcher_3d():
    matcher = HungarianMatcher3D(cls_weight=1, bbox_weight=10)

    # Dummy predicted boxes: 3 predictions
    pred_bboxes = torch.tensor([
        [[0, 0, 0]] * 8,         # close to GT1
        [[10, 10, 10]] * 8,      # far from any GT
        [[1, 1, 1]] * 8          # close to GT2
    ], dtype=torch.float32)

    # Dummy raw logits (not softmaxed)
    pred_logits = torch.tensor([
        [0.1, 2.0],   # high confidence object
        [0.5, 0.5],   # uncertain
        [0.2, 1.5],   # good confidence object
    ], dtype=torch.float32)

    # Ground truth boxes: 2 GTs
    gt_bboxes = torch.tensor([
        [[0.1, 0.1, 0.1]] * 8,  # close to pred_bboxes[0]
        [[1.2, 1.1, 1.3]] * 8   # close to pred_bboxes[2]
    ], dtype=torch.float32)

    gt_labels = torch.tensor([1, 1], dtype=torch.int64)  # both objects

    outputs = {
        "pred_bboxes": pred_bboxes.unsqueeze(0),  # batch size 1
        "pred_logits": pred_logits.unsqueeze(0)   # batch size 1
    }

    targets = gt_bboxes,

    indices = matcher.match(outputs, targets)
    pred_idx, tgt_idx = indices[0]

    print("Matched prediction indices:", pred_idx.tolist())
    print("Matched ground truth indices:", tgt_idx.tolist())

    # Check that two matches found
    assert pred_idx.shape[0] == tgt_idx.shape[0] == 2, "Should match 2 predictions to 2 GT boxes"
    # Only predictions 0 and 2 should match because prediction 1 is far and uncertain
    assert set(pred_idx.tolist()) <= {0, 2}, "Only pred 0 and 2 should be matched"
    print("Test passed")

if __name__ == "__main__":
    test_hungarian_matcher_3d() 