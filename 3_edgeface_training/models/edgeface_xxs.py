from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .loralin_conv import LoRaLinConv1x1


MODEL_PRESETS: dict[str, tuple[int, int, int, int]] = {
    "legacy": (32, 64, 128, 256),
    "mid": (48, 96, 192, 384),
    "widened": (72, 144, 288, 576),
    "teacher_assistant": (96, 192, 384, 768),
}


@dataclass(frozen=True)
class EdgeFaceConfig:
    backbone_name: str = "edgeface_xxs"
    embedding_dim: int = 512
    width_preset: str = "legacy"
    stage_channels: tuple[int, int, int, int] | None = None
    rank_ratio: float = 0.6

    def resolved_stage_channels(self) -> tuple[int, int, int, int]:
        if self.stage_channels is not None:
            return self.stage_channels
        if self.width_preset not in MODEL_PRESETS:
            raise ValueError(
                f"Unknown width_preset={self.width_preset!r}. Available presets: {sorted(MODEL_PRESETS)}"
            )
        return MODEL_PRESETS[self.width_preset]

    def to_metadata(self) -> dict[str, object]:
        return {
            "backbone_name": self.backbone_name,
            "embedding_dim": self.embedding_dim,
            "width_preset": self.width_preset,
            "stage_channels": list(self.resolved_stage_channels()),
            "rank_ratio": self.rank_ratio,
        }


def _normalize_embedding_dim(
    embedding_dim: int | None,
    embedding_size: int | None,
) -> int:
    if embedding_dim is None:
        return embedding_size if embedding_size is not None else 512
    if embedding_size is not None and embedding_size != embedding_dim:
        raise ValueError("embedding_dim and embedding_size must match when both are provided.")
    return embedding_dim


def _normalize_stage_channels(stage_channels: Iterable[int] | None) -> tuple[int, int, int, int] | None:
    if stage_channels is None:
        return None
    normalized = tuple(int(channel) for channel in stage_channels)
    if len(normalized) != 4:
        raise ValueError("stage_channels must contain exactly 4 channel widths.")
    if any(channel <= 0 for channel in normalized):
        raise ValueError("stage_channels values must be positive integers.")
    return normalized


def build_edgeface_config(
    *,
    embedding_dim: int | None = None,
    embedding_size: int | None = None,
    width_preset: str = "legacy",
    stage_channels: Iterable[int] | None = None,
    rank_ratio: float = 0.6,
) -> EdgeFaceConfig:
    return EdgeFaceConfig(
        backbone_name="edgeface_xxs",
        embedding_dim=_normalize_embedding_dim(embedding_dim, embedding_size),
        width_preset=width_preset,
        stage_channels=_normalize_stage_channels(stage_channels),
        rank_ratio=rank_ratio,
    )


def build_edgeface_config_from_metadata(checkpoint: dict[str, object] | None) -> EdgeFaceConfig:
    checkpoint = checkpoint or {}
    return build_edgeface_config(
        embedding_dim=int(checkpoint.get("embedding_dim", 512)),
        width_preset=str(checkpoint.get("width_preset", "legacy")),
        stage_channels=checkpoint.get("stage_channels"),
        rank_ratio=float(checkpoint.get("rank_ratio", 0.6)),
    )


class EdgeFaceXXS(nn.Module):
    """
    Backbone gọn cho pipeline huấn luyện.
    Hỗ trợ preset bề rộng mạng và LoRaLin rank ratio để tăng dung lượng mô hình khi cần.
    """

    def __init__(
        self,
        embedding_dim: int | None = None,
        embedding_size: int | None = None,
        width_preset: str = "legacy",
        stage_channels: Iterable[int] | None = None,
        rank_ratio: float = 0.6,
    ) -> None:
        super().__init__()
        self.config = build_edgeface_config(
            embedding_dim=embedding_dim,
            embedding_size=embedding_size,
            width_preset=width_preset,
            stage_channels=stage_channels,
            rank_ratio=rank_ratio,
        )
        channels = self.config.resolved_stage_channels()

        blocks: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            blocks.append(
                LoRaLinConv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    rank_ratio=self.config.rank_ratio,
                )
            )
            in_channels = out_channels
        blocks.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*blocks)
        self.embedding = nn.Linear(channels[-1], self.config.embedding_dim)
        self.embedding_bn = nn.BatchNorm1d(self.config.embedding_dim)

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    @property
    def rank_ratio(self) -> float:
        return self.config.rank_ratio

    @property
    def stage_channels(self) -> tuple[int, int, int, int]:
        return self.config.resolved_stage_channels()

    def get_config(self) -> EdgeFaceConfig:
        return self.config

    def forward(self, x: torch.Tensor, landmarks: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        embeddings = self.embedding_bn(x)
        norms = torch.norm(embeddings, p=2, dim=1)
        return embeddings, norms
