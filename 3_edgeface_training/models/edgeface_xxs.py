from __future__ import annotations

import torch
from torch import nn

from .loralin_conv import LoRaLinConv1x1


class EdgeFaceXXS(nn.Module):
    """
    Backbone gọn để khởi tạo pipeline huấn luyện.
    Có thể thay từng block bằng kiến trúc EdgeFace-XXS thật mà không đổi script.
    """

    def __init__(
        self,
        embedding_dim: int | None = None,
        embedding_size: int | None = None,
    ) -> None:
        super().__init__()
        if embedding_dim is None:
            embedding_dim = embedding_size if embedding_size is not None else 256
        elif embedding_size is not None and embedding_size != embedding_dim:
            raise ValueError("embedding_dim and embedding_size must match when both are provided.")
        self.features = nn.Sequential(
            LoRaLinConv1x1(3, 32, stride=2),
            LoRaLinConv1x1(32, 64, stride=2),
            LoRaLinConv1x1(64, 128, stride=2),
            LoRaLinConv1x1(128, 256, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Linear(256, embedding_dim)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        embeddings = self.embedding_bn(x)
        norms = torch.norm(embeddings, p=2, dim=1)
        return embeddings, norms
