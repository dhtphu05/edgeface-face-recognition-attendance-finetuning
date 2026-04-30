from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.dataset import HierarchicalImageFolder, resolve_dataset_split_dirs
from models.iresnet_adaface_teacher import IResNet101AdaFaceTeacher
from models.model_factory import build_model_from_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build global teacher class centers for AdaDistill.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, default=PROJECT_ROOT / "weights" / "AdaFace_IR101.pt")
    parser.add_argument(
        "--teacher-backbone",
        choices=["ir101_adaface", "edgeface_ta"],
        default="ir101_adaface",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    return parser.parse_args()


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_train_root(dataset_root: Path) -> Path:
    split_dirs = resolve_dataset_split_dirs(dataset_root)
    return split_dirs.get("train", split_dirs.get("all", dataset_root))


def clean_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_state_dict[key.replace("module.", "") if key.startswith("module.") else key] = value
    return cleaned_state_dict


def build_teacher_model(teacher_backbone: str, teacher_weights: Path) -> tuple[torch.nn.Module, int]:
    checkpoint = torch.load(teacher_weights, map_location="cpu")
    if teacher_backbone == "edgeface_ta":
        model, config = build_model_from_metadata(checkpoint if isinstance(checkpoint, dict) else None)
        missing, unexpected = model.load_state_dict(clean_state_dict(checkpoint), strict=False)
        if missing or unexpected:
            raise ValueError(
                f"TA checkpoint mismatch during teacher center build: missing={list(missing[:5])} unexpected={list(unexpected[:5])}"
            )
        return model, config.embedding_dim

    model = IResNet101AdaFaceTeacher(embedding_dim=512)
    model.load_state_dict(checkpoint, strict=True)
    return model, 512


def main() -> None:
    args = parse_args()
    args.teacher_weights = args.teacher_weights.resolve()
    args.output = args.output.resolve()
    device = resolve_device()
    print(f"🚀 Building teacher centers on device: {device}")

    train_root = resolve_train_root(args.dataset_root)
    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    full_dataset = HierarchicalImageFolder(root=train_root, transform=transform)
    class_names = list(full_dataset.classes)
    if args.max_classes is not None:
        class_names = class_names[: args.max_classes]
    dataset = HierarchicalImageFolder(root=train_root, transform=transform, class_names=class_names)

    if args.max_samples_per_class is not None:
        filtered_samples: list[tuple[str, int]] = []
        filtered_targets: list[int] = []
        per_class_counts = {index: 0 for index in range(len(dataset.classes))}
        for sample_path, label in dataset.samples:
            if per_class_counts[label] >= args.max_samples_per_class:
                continue
            filtered_samples.append((sample_path, label))
            filtered_targets.append(label)
            per_class_counts[label] += 1
        dataset.samples = filtered_samples
        dataset.targets = filtered_targets

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    teacher, embedding_dim = build_teacher_model(args.teacher_backbone, args.teacher_weights)
    teacher = teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    centers_sum = torch.zeros(len(dataset.classes), embedding_dim, dtype=torch.float32)
    counts = torch.zeros(len(dataset.classes), dtype=torch.long)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to("cpu")
            embeddings, _ = teacher(images)
            embeddings = F.normalize(embeddings.detach().float().cpu(), p=2, dim=1)
            for embedding, label in zip(embeddings, labels):
                centers_sum[label] += embedding
                counts[label] += 1
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(loader)} processed for center extraction")

    if torch.any(counts == 0):
        missing_classes = [dataset.classes[index] for index, count in enumerate(counts.tolist()) if count == 0]
        raise ValueError(f"Some classes have zero samples in center extraction: {missing_classes[:10]}")

    centers = F.normalize(centers_sum / counts.unsqueeze(1).float(), p=2, dim=1)
    payload = {
        "teacher_weights": str(args.teacher_weights),
        "teacher_backbone": args.teacher_backbone,
        "dataset_root": str(args.dataset_root),
        "train_root": str(train_root),
        "embedding_dim": embedding_dim,
        "class_names": list(dataset.classes),
        "class_to_index": dataset.class_to_idx,
        "counts": counts.tolist(),
        "centers": centers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"✅ Saved teacher centers: {args.output}")


if __name__ == "__main__":
    main()
