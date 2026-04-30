from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.dataset import resolve_dataset_split_dirs
from models.model_factory import build_model_from_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze embedding norm distributions for selected dataset roots.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def resolve_eval_root(dataset_root: Path) -> Path:
    split_dirs = resolve_dataset_split_dirs(dataset_root)
    return split_dirs.get("val") or split_dirs.get("test") or split_dirs.get("train") or split_dirs["all"]


def summarize_norms(norms: np.ndarray) -> dict[str, float]:
    return {
        "count": int(norms.size),
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "min": float(np.min(norms)),
        "p10": float(np.percentile(norms, 10)),
        "p50": float(np.percentile(norms, 50)),
        "p90": float(np.percentile(norms, 90)),
        "max": float(np.max(norms)),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model, config = build_model_from_metadata(checkpoint if isinstance(checkpoint, dict) else None)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "model_config": config.to_metadata(),
        "datasets": {},
    }

    with torch.no_grad():
        for dataset_root in args.dataset_root:
            eval_root = resolve_eval_root(dataset_root)
            dataset = datasets.ImageFolder(root=str(eval_root), transform=transform)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            collected_norms: list[np.ndarray] = []
            for images, _ in dataloader:
                images = images.to(device)
                _, norms = model(images)
                collected_norms.append(norms.detach().cpu().numpy())
            all_norms = np.concatenate(collected_norms)
            summary = summarize_norms(all_norms)
            report["datasets"][str(dataset_root)] = {
                "eval_root": str(eval_root),
                "summary": summary,
            }
            print(f"{dataset_root} -> mean={summary['mean']:.4f} p10={summary['p10']:.4f} p50={summary['p50']:.4f} p90={summary['p90']:.4f}")

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report to {args.report_json}")


if __name__ == "__main__":
    main()
