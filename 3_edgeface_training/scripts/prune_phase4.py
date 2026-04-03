from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.edgeface_xxs import EdgeFaceXXS, build_edgeface_config_from_metadata


def calculate_l1_norm(weight_tensor):
    return torch.sum(torch.abs(weight_tensor), dim=(1, 2, 3))


def apply_structured_pruning(model, prune_ratio=0.01):
    pruned_layers_count = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.groups == 1:
            if "features.0" in name:
                continue

            prune.ln_structured(module, name="weight", amount=prune_ratio, n=1, dim=0)
            prune.remove(module, "weight")
            pruned_layers_count += 1
            print(f"✂️ Đã cắt tỉa {prune_ratio * 100:.2f}% kênh tại lớp: {name}")

    print(f"Tổng cộng đã cắt tỉa {pruned_layers_count} lớp Tích chập.")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4: conservative structured pruning.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "phase3_widened_model_best.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "phase4_pruned_model.pth",
    )
    parser.add_argument("--prune-ratio", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prune_ratio < 0 or args.prune_ratio >= 1:
        raise ValueError("prune_ratio must be in the range [0, 1).")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"🚀 Bắt đầu quá trình Cắt tỉa (Pruning) trên thiết bị: {device}")

    checkpoint = torch.load(args.input, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    config = build_edgeface_config_from_metadata(checkpoint if isinstance(checkpoint, dict) else None)

    model = EdgeFaceXXS(
        embedding_dim=config.embedding_dim,
        width_preset=config.width_preset,
        stage_channels=config.stage_channels,
        rank_ratio=config.rank_ratio,
    ).to(device)
    model.load_state_dict(state_dict)
    print("✅ Đã nạp thành công trọng số gốc từ Giai đoạn 3.")

    total_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Số lượng tham số trước pruning: {total_params_before:,}")

    if args.prune_ratio > 0:
        model = apply_structured_pruning(model, prune_ratio=args.prune_ratio)

    total_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    payload = {
        **(checkpoint if isinstance(checkpoint, dict) else {}),
        "model_state_dict": model.state_dict(),
        "prune_ratio": args.prune_ratio,
        "pruned": args.prune_ratio > 0,
        "params_before_pruning": total_params_before,
        "params_after_pruning": total_params_after,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"💾 Đã lưu mô hình sau khi cắt tỉa tại: {args.output}")


if __name__ == "__main__":
    main()
