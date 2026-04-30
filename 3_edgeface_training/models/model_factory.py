from __future__ import annotations

from typing import Iterable

from .edgeface_hybrid_kprpe import EdgeFaceHybridKPRPE, build_hybrid_edgeface_config_from_metadata
from .edgeface_xxs import EdgeFaceXXS, build_edgeface_config_from_metadata


def build_model(
    *,
    backbone_name: str,
    embedding_dim: int = 512,
    width_preset: str = "legacy",
    stage_channels: Iterable[int] | None = None,
    rank_ratio: float = 0.6,
    attention_heads: int = 4,
    attention_depth: int = 1,
    kprpe_hidden_dim: int = 32,
):
    if backbone_name == "edgeface_hybrid_kprpe":
        return EdgeFaceHybridKPRPE(
            embedding_dim=embedding_dim,
            width_preset=width_preset,
            stage_channels=stage_channels,
            rank_ratio=rank_ratio,
            attention_heads=attention_heads,
            attention_depth=attention_depth,
            kprpe_hidden_dim=kprpe_hidden_dim,
        )
    return EdgeFaceXXS(
        embedding_dim=embedding_dim,
        width_preset=width_preset,
        stage_channels=stage_channels,
        rank_ratio=rank_ratio,
    )


def build_model_from_metadata(checkpoint: dict[str, object] | None):
    checkpoint = checkpoint or {}
    backbone_name = str(checkpoint.get("backbone_name", "edgeface_xxs"))
    if backbone_name == "edgeface_hybrid_kprpe":
        config = build_hybrid_edgeface_config_from_metadata(checkpoint)
        model = EdgeFaceHybridKPRPE(
            embedding_dim=config.embedding_dim,
            width_preset=config.width_preset,
            stage_channels=config.stage_channels,
            rank_ratio=config.rank_ratio,
            attention_heads=config.attention_heads,
            attention_depth=config.attention_depth,
            kprpe_hidden_dim=config.kprpe_hidden_dim,
        )
        return model, config

    config = build_edgeface_config_from_metadata(checkpoint)
    model = EdgeFaceXXS(
        embedding_dim=config.embedding_dim,
        width_preset=config.width_preset,
        stage_channels=config.stage_channels,
        rank_ratio=config.rank_ratio,
    )
    return model, config
