import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaFaceLoss(nn.Module):
    def __init__(self, embedding_size=512, num_classes=1000, margin=0.4, scale=64.0, h=1.0):
        super(AdaFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.h = h
        
        # Ma trận trọng số đại diện cho các Identity (Học sinh)
        self.weights = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weights)
        
        # Sử dụng BatchNorm1d(affine=False) như một cơ chế tính EMA tự động cho Norm
        self.batch_norm = nn.BatchNorm1d(1, affine=False, momentum=0.01)

    def _apply_adaptive_margin(self, embeddings, norms, labels, normalized_weights):
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        cosine = F.linear(normalized_embeddings, normalized_weights)
        norms_ema = self.batch_norm(norms.unsqueeze(1)).squeeze(1)
        quality_indicator = torch.clamp(norms_ema * self.h, -1.0, 1.0)
        g_angular = self.margin * quality_indicator
        g_additive = self.margin + (self.margin * quality_indicator * -1)
        cosine_m = torch.cos(torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)) + g_angular.unsqueeze(1)) - g_additive.unsqueeze(1)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = (one_hot * cosine_m) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output

    def logits(self, embeddings, norms, labels):
        normalized_weights = F.normalize(self.weights, p=2, dim=1)
        return self._apply_adaptive_margin(embeddings, norms, labels, normalized_weights)

    def logits_partial(
        self,
        embeddings,
        norms,
        labels,
        sample_rate=0.10,
        min_negatives=2048,
        generator: torch.Generator | None = None,
    ):
        positive_classes = torch.unique(labels.detach())
        positive_count = int(positive_classes.numel())
        if positive_count >= self.num_classes:
            return self.logits(embeddings, norms, labels), labels

        negative_candidates = self.num_classes - positive_count
        sampled_negatives = max(min_negatives, int(self.num_classes * sample_rate))
        sampled_negatives = min(sampled_negatives, negative_candidates)
        if sampled_negatives <= 0:
            return self.logits(embeddings, norms, labels), labels

        device = labels.device
        all_classes = torch.arange(self.num_classes, device=device)
        positive_mask = torch.zeros(self.num_classes, dtype=torch.bool, device=device)
        positive_mask[positive_classes] = True
        negative_pool = all_classes[~positive_mask]

        if sampled_negatives >= negative_pool.numel():
            sampled_negative_classes = negative_pool
        else:
            permutation = torch.randperm(negative_pool.numel(), generator=generator)
            sampled_negative_classes = negative_pool[permutation[:sampled_negatives].to(device)]

        subset_classes = torch.cat([positive_classes, sampled_negative_classes], dim=0)
        label_map = torch.full((self.num_classes,), -1, dtype=torch.long, device=device)
        label_map[subset_classes] = torch.arange(subset_classes.numel(), dtype=torch.long, device=device)
        remapped_labels = label_map[labels]
        if torch.any(remapped_labels < 0):
            raise ValueError("Partial FC label remapping failed.")

        normalized_weights = F.normalize(self.weights[subset_classes], p=2, dim=1)
        partial_logits = self._apply_adaptive_margin(embeddings, norms, remapped_labels, normalized_weights)
        return partial_logits, remapped_labels

    def forward(
        self,
        embeddings,
        norms,
        labels,
        classifier_mode="full",
        partial_fc_sample_rate=0.10,
        partial_fc_min_negatives=2048,
        partial_fc_generator: torch.Generator | None = None,
    ):
        if classifier_mode == "partial_fc":
            logits, remapped_labels = self.logits_partial(
                embeddings,
                norms,
                labels,
                sample_rate=partial_fc_sample_rate,
                min_negatives=partial_fc_min_negatives,
                generator=partial_fc_generator,
            )
            return F.cross_entropy(logits, remapped_labels)
        return F.cross_entropy(self.logits(embeddings, norms, labels), labels)
