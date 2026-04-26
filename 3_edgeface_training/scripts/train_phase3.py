from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_losses.adaface_loss import AdaFaceLoss
from core_losses.kd_loss import EmbeddingKDLoss
from dataloaders.dataset import HierarchicalImageFolder, resolve_dataset_split_dirs
from models.edgeface_xxs import MODEL_PRESETS, EdgeFaceXXS
from models.resnet101_teacher import ResNet101Teacher


def compute_classification_logits(embeddings: torch.Tensor, classifier_weights: torch.Tensor) -> torch.Tensor:
    normalized_embeddings = F.normalize(embeddings.float(), p=2, dim=1)
    normalized_weights = F.normalize(classifier_weights.float(), p=2, dim=1)
    return F.linear(normalized_embeddings, normalized_weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 training for EdgeFace student model.")
    parser.add_argument("--dataset-root", type=Path, default=WORKSPACE_ROOT / "2_face_dataset")
    parser.add_argument("--checkpoints-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kd-alpha", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width-preset", choices=sorted(MODEL_PRESETS), default="widened")
    parser.add_argument(
        "--stage-channels",
        type=int,
        nargs=4,
        default=None,
        metavar=("C1", "C2", "C3", "C4"),
        help="Override model width preset with explicit stage channels.",
    )
    parser.add_argument("--rank-ratio", type=float, default=0.7)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-train-batches-per-epoch",
        type=int,
        default=None,
        help="Limit the number of training batches processed in each epoch. Useful for Colab smoke runs on huge datasets.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Limit the number of validation batches processed in each epoch.",
    )
    parser.add_argument(
        "--output-prefix",
        default="phase3_widened_model",
        help="Prefix for checkpoint and metrics files inside checkpoints-dir.",
    )
    parser.add_argument(
        "--student-weights",
        type=Path,
        default=PROJECT_ROOT / "weights" / "edgeface_xxs.pt",
        help="Optional bootstrap weights for the student.",
    )
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=PROJECT_ROOT / "weights" / "resnet101_adaface.pt",
        help="Teacher checkpoint weights.",
    )
    parser.add_argument(
        "--skip-student-bootstrap",
        action="store_true",
        help="Do not load student bootstrap weights.",
    )
    parser.add_argument(
        "--skip-teacher-bootstrap",
        action="store_true",
        help="Do not load teacher checkpoint weights.",
    )
    parser.add_argument(
        "--teacher-pretrained-imagenet",
        action="store_true",
        help="Instantiate the teacher with torchvision ImageNet weights instead of loading teacher checkpoint.",
    )
    parser.add_argument(
        "--strict-bootstrap-check",
        action="store_true",
        help="Abort if bootstrap checkpoint mismatch exceeds configured thresholds.",
    )
    parser.add_argument(
        "--max-missing-ratio",
        type=float,
        default=0.10,
        help="Maximum allowed missing key ratio when strict bootstrap check is enabled.",
    )
    parser.add_argument(
        "--max-unexpected-ratio",
        type=float,
        default=0.10,
        help="Maximum allowed unexpected key ratio when strict bootstrap check is enabled.",
    )
    return parser.parse_args()


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
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
        cleaned_key = key.replace("module.", "") if key.startswith("module.") else key
        cleaned_state_dict[cleaned_key] = value
    return cleaned_state_dict


def maybe_load_bootstrap_weights(
    *,
    model: torch.nn.Module,
    weight_path: Path,
    strict: bool,
    max_missing_ratio: float,
    max_unexpected_ratio: float,
    label: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    report: dict[str, Any] = {
        "enabled": True,
        "path": str(weight_path),
        "loaded": False,
        "strict": strict,
    }
    if not weight_path.exists():
        message = f"Không tìm thấy weights cho {label}: {weight_path}"
        if strict:
            raise FileNotFoundError(message)
        print(f"⚠️ {message}")
        report["error"] = message
        return model, report

    print(f"📥 Đang nạp bootstrap weights cho {label} từ: {weight_path}")
    checkpoint = torch.load(weight_path, map_location="cpu")
    cleaned_state_dict = clean_state_dict(checkpoint)

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    target_key_count = max(1, len(model.state_dict()))
    missing_ratio = len(missing) / target_key_count
    unexpected_ratio = len(unexpected) / target_key_count
    report.update(
        {
            "loaded": True,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "missing_ratio": missing_ratio,
            "unexpected_ratio": unexpected_ratio,
            "sample_missing_keys": list(missing[:5]),
            "sample_unexpected_keys": list(unexpected[:5]),
        }
    )

    print(
        f"✅ Bootstrap load for {label}: missing_keys={len(missing)} "
        f"unexpected_keys={len(unexpected)} missing_ratio={missing_ratio:.3f} "
        f"unexpected_ratio={unexpected_ratio:.3f}"
    )

    if strict and (missing_ratio > max_missing_ratio or unexpected_ratio > max_unexpected_ratio):
        raise ValueError(
            f"{label} bootstrap mismatch too large: missing_ratio={missing_ratio:.3f}, "
            f"unexpected_ratio={unexpected_ratio:.3f}"
        )
    if missing or unexpected:
        print(
            f"⚠️ {label} bootstrap is not an exact match. "
            f"sample_missing={report['sample_missing_keys']} sample_unexpected={report['sample_unexpected_keys']}"
        )
    return model, report


def _dataset_classes(dataset: Dataset) -> list[str]:
    if isinstance(dataset, Subset):
        return _dataset_classes(dataset.dataset)
    if hasattr(dataset, "classes"):
        return list(dataset.classes)
    raise TypeError(f"Unsupported dataset type for class extraction: {type(dataset)!r}")


def _dataset_targets(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        base_targets = _dataset_targets(dataset.dataset)
        return [base_targets[index] for index in dataset.indices]
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    raise TypeError(f"Unsupported dataset type for target extraction: {type(dataset)!r}")


def dataset_label_counts(dataset: Dataset) -> dict[str, int]:
    class_names = _dataset_classes(dataset)
    counts = {class_name: 0 for class_name in class_names}
    for target in _dataset_targets(dataset):
        counts[class_names[target]] += 1
    return counts


def build_datasets(
    dataset_root: Path,
    val_split: float,
    seed: int,
) -> tuple[Dataset, Dataset, list[str], dict[str, Any]]:
    split_dirs = resolve_dataset_split_dirs(dataset_root)
    dataset_structure = "structured" if "train" in split_dirs else "flat"

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    if "train" in split_dirs:
        train_root = split_dirs["train"]
        val_root = split_dirs.get("val", split_dirs["train"])
        train_dataset_full = HierarchicalImageFolder(root=train_root, transform=train_transform)
        train_class_names = list(train_dataset_full.classes)
        val_dataset_full = HierarchicalImageFolder(
            root=val_root,
            transform=val_transform,
            class_names=train_class_names if val_root != train_root else None,
        )
        if val_root != train_root and train_class_names != val_dataset_full.classes:
            raise ValueError(
                "Train/val class mapping mismatch. "
                f"train_classes={train_class_names} val_classes={val_dataset_full.classes}"
            )
        class_names = list(train_class_names)
        if val_root == train_root:
            total_samples = len(train_dataset_full)
            val_size = max(1, int(total_samples * val_split))
            train_size = total_samples - val_size
            if train_size <= 0:
                raise ValueError("Dataset quá nhỏ để tách validation.")
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(total_samples, generator=generator).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            train_dataset = Subset(train_dataset_full, train_indices)
            val_dataset = Subset(val_dataset_full, val_indices)
            dataset_structure = "structured_train_only_random_val_split"
        else:
            train_dataset = train_dataset_full
            val_dataset = val_dataset_full
            dataset_structure = "structured_train_val"
    else:
        train_dataset_full = HierarchicalImageFolder(root=dataset_root, transform=train_transform)
        val_dataset_full = HierarchicalImageFolder(root=dataset_root, transform=val_transform)
        class_names = list(train_dataset_full.classes)
        total_samples = len(train_dataset_full)
        val_size = max(1, int(total_samples * val_split))
        train_size = total_samples - val_size
        if train_size <= 0:
            raise ValueError("Dataset quá nhỏ để tách validation.")
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_samples, generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_dataset = Subset(train_dataset_full, train_indices)
        val_dataset = Subset(val_dataset_full, val_indices)

    train_class_names = _dataset_classes(train_dataset)
    val_class_names = _dataset_classes(val_dataset)
    diagnostics = {
        "dataset_root": str(dataset_root),
        "dataset_structure": dataset_structure,
        "train_class_names": train_class_names,
        "val_class_names": val_class_names,
        "class_mapping_matches": train_class_names == val_class_names,
        "train_class_counts": dataset_label_counts(train_dataset),
        "val_class_counts": dataset_label_counts(val_dataset),
    }
    return train_dataset, val_dataset, class_names, diagnostics


def format_optional_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def checkpoint_payload(
    model: EdgeFaceXXS,
    epoch: int,
    val_accuracy: float,
    args: argparse.Namespace,
    run_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_accuracy": val_accuracy,
        "kd_alpha": args.kd_alpha,
        "dataset_root": str(args.dataset_root),
        "seed": args.seed,
        "run_diagnostics": run_diagnostics,
    }
    payload.update(model.get_config().to_metadata())
    return payload


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device()
    print(f"🚀 Đang huấn luyện trên thiết bị: {device}")

    if args.kd_alpha < 0:
        raise ValueError("kd_alpha must be >= 0.")
    if args.max_missing_ratio < 0 or args.max_unexpected_ratio < 0:
        raise ValueError("Bootstrap mismatch ratios must be >= 0.")
    if args.max_train_batches_per_epoch is not None and args.max_train_batches_per_epoch <= 0:
        raise ValueError("max_train_batches_per_epoch must be > 0 when provided.")
    if args.max_val_batches is not None and args.max_val_batches <= 0:
        raise ValueError("max_val_batches must be > 0 when provided.")
    if args.kd_alpha > 0 and args.skip_teacher_bootstrap and not args.teacher_pretrained_imagenet:
        raise ValueError(
            "KD requires a compatible teacher. Use --teacher-pretrained-imagenet or do not skip teacher bootstrap."
        )

    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, class_names, dataset_diagnostics = build_datasets(
        dataset_root=args.dataset_root,
        val_split=args.val_split,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    student_model = EdgeFaceXXS(
        embedding_dim=args.embedding_dim,
        width_preset=args.width_preset,
        stage_channels=args.stage_channels,
        rank_ratio=args.rank_ratio,
    )
    student_param_count = sum(parameter.numel() for parameter in student_model.parameters())

    kd_enabled = args.kd_alpha > 0
    teacher_mode = "disabled"
    teacher_model: ResNet101Teacher | None = None
    bootstrap_reports: dict[str, Any] = {
        "student": {
            "enabled": not args.skip_student_bootstrap,
            "path": str(args.student_weights),
        },
        "teacher": {
            "enabled": kd_enabled and not args.skip_teacher_bootstrap and not args.teacher_pretrained_imagenet,
            "path": str(args.teacher_weights),
        },
    }

    if not args.skip_student_bootstrap:
        student_model, bootstrap_reports["student"] = maybe_load_bootstrap_weights(
            model=student_model,
            weight_path=args.student_weights,
            strict=args.strict_bootstrap_check,
            max_missing_ratio=args.max_missing_ratio,
            max_unexpected_ratio=args.max_unexpected_ratio,
            label="student",
        )
    else:
        print("ℹ️ Student bootstrap: disabled by flag.")
        bootstrap_reports["student"] = {"enabled": False, "reason": "skip_student_bootstrap"}

    if kd_enabled:
        teacher_model = ResNet101Teacher(
            embedding_size=args.embedding_dim,
            pretrained=args.teacher_pretrained_imagenet,
        )
        if args.teacher_pretrained_imagenet:
            teacher_mode = "imagenet_pretrained"
            bootstrap_reports["teacher"] = {
                "enabled": False,
                "reason": "teacher_pretrained_imagenet",
            }
            print("ℹ️ Teacher mode: torchvision ImageNet pretrained.")
        else:
            teacher_mode = "checkpoint_bootstrap"
            teacher_model, bootstrap_reports["teacher"] = maybe_load_bootstrap_weights(
                model=teacher_model,
                weight_path=args.teacher_weights,
                strict=args.strict_bootstrap_check,
                max_missing_ratio=args.max_missing_ratio,
                max_unexpected_ratio=args.max_unexpected_ratio,
                label="teacher",
            )
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad = False
    else:
        print("ℹ️ KD: disabled because kd_alpha=0.")
        bootstrap_reports["teacher"] = {"enabled": False, "reason": "kd_disabled"}

    student_model = student_model.to(device)

    run_diagnostics: dict[str, Any] = {
        "dataset": dataset_diagnostics,
        "student_param_count": student_param_count,
        "student_bootstrap": bootstrap_reports["student"],
        "teacher_bootstrap": bootstrap_reports["teacher"],
        "kd_enabled": kd_enabled,
        "kd_alpha": args.kd_alpha,
        "teacher_mode": teacher_mode,
        "device": str(device),
        "output_prefix": args.output_prefix,
        "max_train_batches_per_epoch": args.max_train_batches_per_epoch,
        "max_val_batches": args.max_val_batches,
    }

    print(
        f"📁 Classes: {len(class_names)} | Train: {len(train_dataset)} | Val: {len(val_dataset)} | "
        f"Preset: {args.width_preset} | Rank ratio: {args.rank_ratio} | Params: {student_param_count:,}"
    )
    print(f"🧭 Dataset structure: {dataset_diagnostics['dataset_structure']}")
    print(f"🧭 Train/val class mapping match: {dataset_diagnostics['class_mapping_matches']}")
    print(f"🧭 Train classes: {dataset_diagnostics['train_class_names']}")
    print(f"🧭 Val classes: {dataset_diagnostics['val_class_names']}")
    print(f"🧭 Train counts: {dataset_diagnostics['train_class_counts']}")
    print(f"🧭 Val counts: {dataset_diagnostics['val_class_counts']}")
    print(f"🧭 Student bootstrap enabled: {not args.skip_student_bootstrap}")
    print(f"🧭 KD enabled: {kd_enabled} | Teacher mode: {teacher_mode}")
    print(
        f"🧭 Partial epoch: train_batches="
        f"{args.max_train_batches_per_epoch if args.max_train_batches_per_epoch is not None else 'full'} | "
        f"val_batches={args.max_val_batches if args.max_val_batches is not None else 'full'}"
    )

    adaface_criterion = AdaFaceLoss(embedding_size=args.embedding_dim, num_classes=len(class_names)).to(device)
    kd_criterion = EmbeddingKDLoss(alpha=args.kd_alpha).to(device) if kd_enabled else None
    optimizer = optim.AdamW(
        [
            {"params": student_model.parameters(), "weight_decay": 5e-4},
            {"params": adaface_criterion.parameters(), "weight_decay": 5e-4},
        ],
        lr=args.learning_rate,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    scaler = amp.GradScaler(device_type, enabled=(device_type == "cuda"))

    history: list[dict[str, Any]] = []
    best_val_accuracy = float("-inf")
    best_checkpoint_path = args.checkpoints_dir / f"{args.output_prefix}_best.pth"

    for epoch in range(args.epochs):
        student_model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        train_cosine_sum = 0.0
        train_cosine_batches = 0
        processed_train_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if (
                args.max_train_batches_per_epoch is not None
                and batch_idx >= args.max_train_batches_per_epoch
            ):
                break
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                student_embeddings, student_norms = student_model(images)
                loss_adaface = adaface_criterion(student_embeddings, student_norms, labels)

                teacher_embeddings = None
                if kd_enabled:
                    assert teacher_model is not None and kd_criterion is not None
                    with torch.no_grad():
                        teacher_embeddings, _ = teacher_model(images)
                    loss_kd = kd_criterion(student_embeddings, teacher_embeddings)
                else:
                    loss_kd = torch.zeros((), device=device)

                loss = loss_adaface + loss_kd

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            logits = compute_classification_logits(student_embeddings.detach(), adaface_criterion.weights.detach())
            predictions = logits.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            if kd_enabled and teacher_embeddings is not None:
                batch_cosine = F.cosine_similarity(
                    student_embeddings.detach().float(),
                    teacher_embeddings.detach().float(),
                    dim=1,
                ).mean().item()
                train_cosine_sum += batch_cosine
                train_cosine_batches += 1
            processed_train_batches += 1

            if batch_idx % 10 == 0:
                kd_log = "disabled" if not kd_enabled else f"{loss_kd.item():.4f}"
                train_batch_total = (
                    args.max_train_batches_per_epoch
                    if args.max_train_batches_per_epoch is not None
                    else len(train_loader)
                )
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] | Batch [{batch_idx}/{train_batch_total}] | "
                    f"Loss: {loss.item():.4f} (AdaFace: {loss_adaface.item():.4f}, KD: {kd_log})"
                )

        train_accuracy = 100.0 * train_correct / max(1, train_total)
        train_cosine = (
            train_cosine_sum / max(1, train_cosine_batches) if kd_enabled and train_cosine_batches > 0 else None
        )

        student_model.eval()
        val_correct = 0
        val_total = 0
        val_cosine_sum = 0.0
        val_cosine_batches = 0
        processed_val_batches = 0
        with torch.no_grad():
            for val_batch_idx, (images, labels) in enumerate(val_loader):
                if args.max_val_batches is not None and val_batch_idx >= args.max_val_batches:
                    break
                images = images.to(device)
                labels = labels.to(device)
                student_embeddings, _ = student_model(images)
                logits = compute_classification_logits(student_embeddings, adaface_criterion.weights)
                predictions = logits.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                processed_val_batches += 1

                if kd_enabled:
                    assert teacher_model is not None
                    teacher_embeddings, _ = teacher_model(images)
                    batch_cosine = F.cosine_similarity(
                        student_embeddings.float(),
                        teacher_embeddings.float(),
                        dim=1,
                    ).mean().item()
                    val_cosine_sum += batch_cosine
                    val_cosine_batches += 1

        val_accuracy = 100.0 * val_correct / max(1, val_total)
        val_cosine = (
            val_cosine_sum / max(1, val_cosine_batches) if kd_enabled and val_cosine_batches > 0 else None
        )
        avg_loss = total_loss / max(1, processed_train_batches)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "train_cosine": train_cosine,
                "val_cosine": val_cosine,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "kd_enabled": kd_enabled,
                "processed_train_batches": processed_train_batches,
                "processed_val_batches": processed_val_batches,
            }
        )

        scheduler.step()
        print(
            f"🎉 Hết Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
            f"Val Acc: {val_accuracy:.2f}% | Train Cosine: {format_optional_metric(train_cosine)} | "
            f"Val Cosine: {format_optional_metric(val_cosine)}"
        )

        epoch_checkpoint = args.checkpoints_dir / f"{args.output_prefix}_ep{epoch + 1}.pth"
        payload = checkpoint_payload(student_model, epoch + 1, val_accuracy, args, run_diagnostics)
        torch.save(payload, epoch_checkpoint)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(payload, best_checkpoint_path)

    final_payload = checkpoint_payload(student_model, args.epochs, best_val_accuracy, args, run_diagnostics)
    final_checkpoint_path = args.checkpoints_dir / f"{args.output_prefix}.pth"
    torch.save(final_payload, final_checkpoint_path)

    metrics_path = args.checkpoints_dir / f"{args.output_prefix}_metrics.json"
    metrics_summary = {
        "best_val_accuracy": best_val_accuracy,
        "epochs": args.epochs,
        "history": history,
        "model_config": student_model.get_config().to_metadata(),
        "dataset_diagnostics": dataset_diagnostics,
        "run_diagnostics": run_diagnostics,
    }
    metrics_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    print(f"✅ Đã lưu checkpoint cuối: {final_checkpoint_path}")
    print(f"✅ Đã lưu checkpoint tốt nhất: {best_checkpoint_path}")
    print(f"✅ Đã lưu metrics: {metrics_path}")


if __name__ == "__main__":
    main()
