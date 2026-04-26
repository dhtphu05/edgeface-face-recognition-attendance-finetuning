from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AdaDistillStats:
    adaptive_p: float
    gt_confidence: float
    loss_easy: float
    loss_hard: float
    loss_total: float


class AdaDistillLoss(nn.Module):
    def __init__(self, weight: float = 0.5, scale: float = 64.0) -> None:
        super().__init__()
        self.weight = weight
        self.scale = scale
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        *,
        student_embeddings: torch.Tensor,
        student_classifier_weights: torch.Tensor,
        teacher_centers: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, AdaDistillStats]:
        student_norm = F.normalize(student_embeddings.float(), p=2, dim=1)
        teacher_center_norm = F.normalize(teacher_centers.float(), p=2, dim=1)
        teacher_logits = F.linear(student_norm, teacher_center_norm) * self.scale
        loss_easy = F.cross_entropy(teacher_logits, labels)

        batch_labels = torch.unique(labels, sorted=True)
        student_weight_subset = F.normalize(student_classifier_weights[batch_labels].float(), p=2, dim=1)
        teacher_center_subset = teacher_center_norm[batch_labels]
        student_subset_logits = F.linear(student_norm, student_weight_subset) * self.scale
        teacher_subset_logits = F.linear(student_norm, teacher_center_subset) * self.scale
        loss_hard = self.kl_div(
            F.log_softmax(student_subset_logits, dim=1),
            F.softmax(teacher_subset_logits.detach(), dim=1),
        )

        with torch.no_grad():
            batch_subset_probs = F.softmax(student_subset_logits.detach().float(), dim=1)
            batch_gt_confidence = batch_subset_probs.gather(1, torch.bucketize(labels, batch_labels).view(-1, 1))
            mean_gt_confidence = batch_gt_confidence.mean()
            random_baseline = 1.0 / max(1, int(batch_labels.numel()))
            adaptive_p = ((mean_gt_confidence - random_baseline) / max(1e-6, 1.0 - random_baseline)).clamp(0.0, 1.0)

        kd_loss = ((1.0 - adaptive_p) * loss_easy) + (adaptive_p * loss_hard)
        total_loss = kd_loss * self.weight
        stats = AdaDistillStats(
            adaptive_p=float(adaptive_p.item()),
            gt_confidence=float(mean_gt_confidence.item()),
            loss_easy=float(loss_easy.item()),
            loss_hard=float(loss_hard.item()),
            loss_total=float(total_loss.item()),
        )
        return total_loss, stats
