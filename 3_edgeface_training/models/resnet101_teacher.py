from __future__ import annotations

from torch import nn
from torchvision.models import ResNet101_Weights, resnet101


class ResNet101Teacher(nn.Module):
    def __init__(
        self,
        embedding_dim: int | None = None,
        embedding_size: int | None = None,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim is None:
            embedding_dim = embedding_size if embedding_size is not None else 256
        elif embedding_size is not None and embedding_size != embedding_dim:
            raise ValueError("embedding_dim and embedding_size must match when both are provided.")
        weights = ResNet101_Weights.DEFAULT if pretrained else None
        backbone = resnet101(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        x = self.backbone(x)
        embeddings = self.projection(x)
        norms = embeddings.norm(p=2, dim=1)
        return embeddings, norms
