from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .edgeface_xxs import MODEL_PRESETS, _normalize_embedding_dim, _normalize_stage_channels
from .loralin_conv import LoRaLinConv1x1


@dataclass(frozen=True)
class HybridEdgeFaceConfig:
    backbone_name: str = "edgeface_hybrid_kprpe"
    embedding_dim: int = 512
    width_preset: str = "widened"
    stage_channels: tuple[int, int, int, int] | None = None
    rank_ratio: float = 0.7
    attention_heads: int = 4
    attention_depth: int = 1
    kprpe_hidden_dim: int = 32

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
            "attention_heads": self.attention_heads,
            "attention_depth": self.attention_depth,
            "kprpe_hidden_dim": self.kprpe_hidden_dim,
        }


def build_hybrid_edgeface_config(
    *,
    embedding_dim: int | None = None,
    embedding_size: int | None = None,
    width_preset: str = "widened",
    stage_channels: Iterable[int] | None = None,
    rank_ratio: float = 0.7,
    attention_heads: int = 4,
    attention_depth: int = 1,
    kprpe_hidden_dim: int = 32,
) -> HybridEdgeFaceConfig:
    return HybridEdgeFaceConfig(
        embedding_dim=_normalize_embedding_dim(embedding_dim, embedding_size),
        width_preset=width_preset,
        stage_channels=_normalize_stage_channels(stage_channels),
        rank_ratio=rank_ratio,
        attention_heads=attention_heads,
        attention_depth=attention_depth,
        kprpe_hidden_dim=kprpe_hidden_dim,
    )


def build_hybrid_edgeface_config_from_metadata(checkpoint: dict[str, object] | None) -> HybridEdgeFaceConfig:
    checkpoint = checkpoint or {}
    return build_hybrid_edgeface_config(
        embedding_dim=int(checkpoint.get("embedding_dim", 512)),
        width_preset=str(checkpoint.get("width_preset", "widened")),
        stage_channels=checkpoint.get("stage_channels"),
        rank_ratio=float(checkpoint.get("rank_ratio", 0.7)),
        attention_heads=int(checkpoint.get("attention_heads", 4)),
        attention_depth=int(checkpoint.get("attention_depth", 1)),
        kprpe_hidden_dim=int(checkpoint.get("kprpe_hidden_dim", 32)),
    )


class LandmarkRelativeBias(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(
        self,
        landmarks: torch.Tensor | None,
        spatial_size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if landmarks is None:
            return None

        if landmarks.ndim != 3 or landmarks.shape[1:] != (5, 2):
            raise ValueError(f"Expected landmarks shape [B, 5, 2], got {tuple(landmarks.shape)}")

        landmarks = landmarks.to(device=device, dtype=dtype)
        if torch.max(torch.abs(landmarks)) > 1.5:
            landmarks = landmarks / 112.0
        landmarks = torch.clamp(landmarks, 0.0, 1.0)

        batch_size = landmarks.size(0)
        height, width = spatial_size
        y_coords = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype)
        x_coords = torch.linspace(0.0, 1.0, steps=width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        token_grid = torch.stack([grid_x, grid_y], dim=-1).view(1, height * width, 1, 2)

        expanded_landmarks = landmarks.unsqueeze(1)
        distances = torch.norm(token_grid - expanded_landmarks, dim=-1)
        min_distances, _ = distances.min(dim=-1)
        mean_distances = distances.mean(dim=-1)
        affinity = torch.exp(-min_distances * 6.0)
        features = torch.stack([affinity, min_distances, mean_distances], dim=-1)
        token_bias = self.mlp(features).permute(0, 2, 1)
        pairwise_bias = -torch.abs(token_bias.unsqueeze(-1) - token_bias.unsqueeze(-2))
        return pairwise_bias


class STDABlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, kprpe_hidden_dim: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"Attention dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.landmark_bias = LandmarkRelativeBias(num_heads=num_heads, hidden_dim=kprpe_hidden_dim)

    def forward(self, x: torch.Tensor, spatial_size: tuple[int, int], landmarks: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, token_count, channels = x.shape
        height, width = spatial_size
        residual = x

        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).view(batch_size, token_count, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        bias = self.landmark_bias(landmarks, spatial_size, x.device, x.dtype)
        if bias is not None:
            attn = attn + bias
        attn = F.softmax(attn, dim=-1)
        x = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, token_count, channels)
        x = self.proj(x)
        x = residual + x

        conv_residual = x
        feature_map = x.transpose(1, 2).reshape(batch_size, channels, height, width)
        feature_map = self.dw_conv(feature_map)
        x = conv_residual + feature_map.flatten(2).transpose(1, 2)
        x = x + self.mlp(self.norm2(x))
        return x


class EdgeFaceHybridKPRPE(nn.Module):
    def __init__(
        self,
        embedding_dim: int | None = None,
        embedding_size: int | None = None,
        width_preset: str = "widened",
        stage_channels: Iterable[int] | None = None,
        rank_ratio: float = 0.7,
        attention_heads: int = 4,
        attention_depth: int = 1,
        kprpe_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.config = build_hybrid_edgeface_config(
            embedding_dim=embedding_dim,
            embedding_size=embedding_size,
            width_preset=width_preset,
            stage_channels=stage_channels,
            rank_ratio=rank_ratio,
            attention_heads=attention_heads,
            attention_depth=attention_depth,
            kprpe_hidden_dim=kprpe_hidden_dim,
        )
        channels = self.config.resolved_stage_channels()

        self.stage1 = LoRaLinConv1x1(3, channels[0], stride=2, rank_ratio=self.config.rank_ratio)
        self.stage2 = LoRaLinConv1x1(channels[0], channels[1], stride=2, rank_ratio=self.config.rank_ratio)
        self.stage3_down = LoRaLinConv1x1(channels[1], channels[2], stride=2, rank_ratio=self.config.rank_ratio)
        self.stage4_down = LoRaLinConv1x1(channels[2], channels[3], stride=2, rank_ratio=self.config.rank_ratio)

        self.stage3_blocks = nn.ModuleList(
            [
                STDABlock(
                    dim=channels[2],
                    num_heads=self.config.attention_heads,
                    kprpe_hidden_dim=self.config.kprpe_hidden_dim,
                )
                for _ in range(self.config.attention_depth)
            ]
        )
        self.stage4_blocks = nn.ModuleList(
            [
                STDABlock(
                    dim=channels[3],
                    num_heads=self.config.attention_heads,
                    kprpe_hidden_dim=self.config.kprpe_hidden_dim,
                )
                for _ in range(self.config.attention_depth)
            ]
        )

        self.embedding = nn.Linear(channels[3], self.config.embedding_dim)
        self.embedding_bn = nn.BatchNorm1d(self.config.embedding_dim)

    def get_config(self) -> HybridEdgeFaceConfig:
        return self.config

    def _run_attention_stage(
        self,
        x: torch.Tensor,
        blocks: nn.ModuleList,
        landmarks: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        for block in blocks:
            tokens = block(tokens, spatial_size=(height, width), landmarks=landmarks)
        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)

    def forward(self, x: torch.Tensor, landmarks: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3_down(x)
        x = self._run_attention_stage(x, self.stage3_blocks, landmarks)
        x = self.stage4_down(x)
        x = self._run_attention_stage(x, self.stage4_blocks, landmarks)
        x = x.mean(dim=(2, 3))
        x = self.embedding(x)
        embeddings = self.embedding_bn(x)
        norms = torch.norm(embeddings, p=2, dim=1)
        return embeddings, norms
