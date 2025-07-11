import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from vision3d.utils.build import build_utils
from vision3d.utils.registry import LOSSES

@LOSSES.register()
class MultiLoss3D(nn.Module):
    def __init__(self, matcher_cfg: dict, weight_dict=None):
        super().__init__()
        self.matcher = build_utils(matcher_cfg)
        self.weight_dict = weight_dict or {'loss_bbox': 5, 'loss_ce': 1}
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """
        outputs: dict with keys 'pred_bboxes' (BxNx8x3) and 'pred_confidence' (BxNx2)
        targets: list of dicts, each with 'boxes' (Mx8x3) and 'labels' (M,)
        """
        indices = self.matcher.match(outputs, targets)

        loss_bbox = 0.0
        loss_ce = 0.0

        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            pred_boxes = outputs['pred_bbox3d'][batch_idx][pred_idx]  # (num_matched,8,3)
            tgt_boxes = targets[batch_idx][tgt_idx]          # (num_matched,8,3)
            loss_bbox += F.l1_loss(pred_boxes, tgt_boxes, reduction='sum')

            pred_logits = outputs['pred_logits'][batch_idx]       # (num_preds, 2)
            target_classes = torch.zeros(pred_logits.size(0), dtype=torch.long, device=pred_logits.device)
            target_classes[pred_idx] = 1  # matched preds are positive class
            loss_ce += self.ce_loss(pred_logits, target_classes)

        # normalize bbox loss by batch size (optional, can also normalize by number matched)
        batch_size = len(targets)
        loss_bbox = loss_bbox / batch_size
        loss_ce = loss_ce / batch_size

        losses = {
            'loss_bbox': loss_bbox,
            'loss_ce': loss_ce,
        }
        losses['total_loss'] = self.weight_dict['loss_bbox'] * loss_bbox + self.weight_dict['loss_ce'] * loss_ce
        return losses