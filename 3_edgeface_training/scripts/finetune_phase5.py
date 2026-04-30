from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_losses import AdaFaceLoss
from dataloaders import build_dataloaders
from models.model_factory import build_model_from_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5: finetune pruned model.")
    parser.add_argument("--dataset-root", type=Path, default=WORKSPACE_ROOT / "2_face_dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "phase4_pruned_model.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "phase5_final_model.pth",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    checkpoint = torch.load(args.input, map_location="cpu")
    train_loader, _, class_names = build_dataloaders(
        args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model, config = build_model_from_metadata(checkpoint if isinstance(checkpoint, dict) else None)
    model = model.to(device)
    model.load_state_dict(state_dict)
    criterion = AdaFaceLoss(embedding_size=config.embedding_dim, num_classes=len(class_names)).to(device)
    optimizer = AdamW(list(model.parameters()) + list(criterion.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"finetune {epoch + 1}/{args.epochs}", leave=False)
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings, norms = model(images)
            loss = criterion(embeddings, norms, labels)
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **(checkpoint if isinstance(checkpoint, dict) else {}),
        "model_state_dict": model.state_dict(),
        "finetuned_epochs": args.epochs,
    }
    payload.update(config.to_metadata())
    torch.save(payload, args.output)
    print(f"Saved finetuned model to {args.output}")


if __name__ == "__main__":
    main()
