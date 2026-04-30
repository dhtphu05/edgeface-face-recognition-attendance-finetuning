from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch import amp
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_losses.adaface_loss import AdaFaceLoss
from core_losses.adadistill_loss import AdaDistillLoss, AdaDistillStats
from core_losses.kd_loss import EmbeddingKDLoss
from dataloaders.dataset import HierarchicalImageFolder, resolve_dataset_split_dirs
from models.domain_adversarial import DomainDiscriminator, GradientReversalLayer
from models.edgeface_xxs import MODEL_PRESETS
from models.iresnet_adaface_teacher import IResNet101AdaFaceTeacher
from models.model_factory import build_model, build_model_from_metadata
from models.resnet101_teacher import ResNet101Teacher


def compute_classification_logits(embeddings: torch.Tensor, classifier_weights: torch.Tensor) -> torch.Tensor:
    normalized_embeddings = F.normalize(embeddings.float(), p=2, dim=1)
    normalized_weights = F.normalize(classifier_weights.float(), p=2, dim=1)
    return F.linear(normalized_embeddings, normalized_weights)


def prepare_teacher_inputs(images: torch.Tensor, teacher_backbone: str) -> torch.Tensor:
    if teacher_backbone == "ir101_adaface":
        return F.interpolate(images.float(), size=(112, 112), mode="bilinear", align_corners=False)
    return images


class DirectionalMotionBlur:
    def __init__(self, probability: float, kernel_size: int) -> None:
        self.probability = probability
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    def __call__(self, image):
        if self.probability <= 0 or random.random() >= self.probability:
            return image

        direction = random.choice(["horizontal", "vertical", "diag_down", "diag_up"])
        kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
        if direction == "horizontal":
            row = self.kernel_size // 2
            for column in range(self.kernel_size):
                kernel[row, column] = 1.0 / self.kernel_size
        elif direction == "vertical":
            column = self.kernel_size // 2
            for row in range(self.kernel_size):
                kernel[row, column] = 1.0 / self.kernel_size
        elif direction == "diag_down":
            for index in range(self.kernel_size):
                kernel[index, index] = 1.0 / self.kernel_size
        else:
            for index in range(self.kernel_size):
                column = self.kernel_size - 1 - index
                kernel[index, column] = 1.0 / self.kernel_size

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0)
        kernel_tensor = torch.from_numpy(kernel).view(1, 1, self.kernel_size, self.kernel_size)
        kernel_tensor = kernel_tensor.repeat(3, 1, 1, 1)
        padding = self.kernel_size // 2
        blurred = F.conv2d(
            F.pad(image_tensor, (padding, padding, padding, padding), mode="reflect"),
            kernel_tensor,
            groups=3,
        )
        blurred = blurred.squeeze(0).clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
        return Image.fromarray((blurred * 255.0).astype(np.uint8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 training for EdgeFace student model.")
    parser.add_argument("--dataset-root", type=Path, default=WORKSPACE_ROOT / "2_face_dataset")
    parser.add_argument("--checkpoints-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument(
        "--backbone",
        choices=["edgeface_xxs", "edgeface_hybrid_kprpe"],
        default="edgeface_xxs",
        help="Student backbone family.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kd-alpha", type=float, default=250.0)
    parser.add_argument(
        "--kd-mode",
        choices=["none", "embedding_mse", "adadistill"],
        default="embedding_mse",
        help="KD strategy. Use 'none' or kd_alpha=0 to preserve AdaFace-only training.",
    )
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
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-depth", type=int, default=1)
    parser.add_argument("--kprpe-hidden-dim", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--landmark-metadata-root", type=Path, default=None)
    parser.add_argument("--motion-blur-prob", type=float, default=0.0)
    parser.add_argument("--motion-blur-kernel-size", type=int, default=7)
    parser.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Restrict dataset loading to the first N sorted classes. Useful for subset-first AdaDistill rollouts.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Cap the number of samples per class before train/val splitting.",
    )
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
        "--teacher-backbone",
        choices=["resnet101_imagenet", "ir101_adaface", "edgeface_ta"],
        default="resnet101_imagenet",
        help="Teacher architecture. AdaDistill faithful branch should use ir101_adaface.",
    )
    parser.add_argument(
        "--teacher-centers-path",
        type=Path,
        default=None,
        help="Precomputed teacher class centers required by kd-mode=adadistill.",
    )
    parser.add_argument(
        "--adadistill-weight",
        type=float,
        default=0.5,
        help="Weight applied to the AdaDistill KD term.",
    )
    parser.add_argument("--adaface-margin", type=float, default=0.4)
    parser.add_argument("--adaface-scale", type=float, default=64.0)
    parser.add_argument("--adaface-h", type=float, default=1.0)
    parser.add_argument(
        "--classifier-mode",
        choices=["full", "partial_fc"],
        default="full",
        help="Classifier head mode. Partial FC is intended for large public-ID warmup only.",
    )
    parser.add_argument("--partial-fc-sample-rate", type=float, default=0.10)
    parser.add_argument("--partial-fc-min-negatives", type=int, default=2048)
    parser.add_argument("--partial-fc-seed", type=int, default=1234)
    parser.add_argument(
        "--teacher-logit-scale",
        type=float,
        default=64.0,
        help="Scale used when computing teacher-center logits in AdaDistill mode.",
    )
    parser.add_argument(
        "--domain-adversarial",
        action="store_true",
        help="Enable domain-adversarial refinement during training.",
    )
    parser.add_argument(
        "--domain-loss-weight",
        type=float,
        default=0.05,
        help="Weight applied to the domain adversarial loss term.",
    )
    parser.add_argument(
        "--domain-label-source",
        choices=["session", "split_origin", "capture_condition"],
        default="session",
        help="Source used to derive domain labels for adversarial refinement.",
    )
    parser.add_argument(
        "--domain-session-gap-ms",
        type=int,
        default=3000,
        help="Session boundary gap when deriving session-based domain labels from timestamps.",
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
    if isinstance(dataset, (LandmarkMetadataDataset, DomainLabeledDataset)):
        return _dataset_classes(dataset.base_dataset)
    if hasattr(dataset, "classes"):
        return list(dataset.classes)
    raise TypeError(f"Unsupported dataset type for class extraction: {type(dataset)!r}")


def _dataset_targets(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        base_targets = _dataset_targets(dataset.dataset)
        return [base_targets[index] for index in dataset.indices]
    if isinstance(dataset, LandmarkMetadataDataset):
        return _dataset_targets(dataset.base_dataset)
    if isinstance(dataset, DomainLabeledDataset):
        return _dataset_targets(dataset.base_dataset)
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    raise TypeError(f"Unsupported dataset type for target extraction: {type(dataset)!r}")


def dataset_label_counts(dataset: Dataset) -> dict[str, int]:
    class_names = _dataset_classes(dataset)
    counts = {class_name: 0 for class_name in class_names}
    for target in _dataset_targets(dataset):
        counts[class_names[target]] += 1
    return counts


def maybe_limit_hierarchical_dataset(
    dataset: HierarchicalImageFolder,
    *,
    max_samples_per_class: int | None,
) -> HierarchicalImageFolder:
    if max_samples_per_class is None:
        return dataset
    filtered_samples: list[tuple[str, int]] = []
    filtered_targets: list[int] = []
    per_class_counts = {index: 0 for index in range(len(dataset.classes))}
    for image_path, label in dataset.samples:
        if per_class_counts[label] >= max_samples_per_class:
            continue
        filtered_samples.append((image_path, label))
        filtered_targets.append(label)
        per_class_counts[label] += 1
    dataset.samples = filtered_samples
    dataset.targets = filtered_targets
    if not dataset.samples:
        raise ValueError("No samples remain after applying max_samples_per_class.")
    return dataset


def _dataset_paths(dataset: Dataset) -> list[Path]:
    if isinstance(dataset, Subset):
        base_paths = _dataset_paths(dataset.dataset)
        return [base_paths[index] for index in dataset.indices]
    if isinstance(dataset, LandmarkMetadataDataset):
        return list(dataset.paths)
    if isinstance(dataset, DomainLabeledDataset):
        return _dataset_paths(dataset.base_dataset)
    if hasattr(dataset, "samples"):
        return [Path(sample_path) for sample_path, _ in dataset.samples]
    raise TypeError(f"Unsupported dataset type for path extraction: {type(dataset)!r}")


def extract_timestamp_from_path(path: Path) -> int | None:
    digit_tokens = re.findall(r"\d{6,}", path.stem)
    if not digit_tokens:
        return None
    return int(max(digit_tokens, key=len))


def extract_capture_condition(path: Path) -> str:
    suffix = path.stem.split("_")[-1].lower()
    known = {"straight", "left", "right", "up", "down"}
    return suffix if suffix in known else "unknown"


def _session_group_keys(paths: list[Path], session_gap_ms: int) -> dict[Path, str]:
    keyed_paths: list[tuple[tuple[int, str], Path]] = []
    untimed_paths: list[Path] = []
    for path in paths:
        timestamp = extract_timestamp_from_path(path)
        if timestamp is None:
            untimed_paths.append(path)
        else:
            keyed_paths.append(((timestamp, path.name), path))

    keyed_paths.sort(key=lambda item: item[0])
    labels: dict[Path, str] = {}
    session_index = -1
    last_timestamp: int | None = None
    for (timestamp, _), path in keyed_paths:
        if last_timestamp is None or timestamp - last_timestamp > session_gap_ms:
            session_index += 1
        labels[path] = f"session_{session_index}"
        last_timestamp = timestamp

    for path in untimed_paths:
        prefix = "_".join(path.stem.split("_")[:2]) or path.stem
        labels[path] = f"untimed_{prefix}"
    return labels


def resolve_domain_keys(
    dataset: Dataset,
    *,
    dataset_root: Path,
    domain_label_source: str,
    session_gap_ms: int,
) -> tuple[list[str], str]:
    paths = _dataset_paths(dataset)

    def split_origin_label(path: Path) -> str:
        try:
            relative = path.relative_to(dataset_root)
            return relative.parts[0] if relative.parts else "unknown"
        except ValueError:
            return path.parts[-3] if len(path.parts) >= 3 else "unknown"

    session_keys = _session_group_keys(paths, session_gap_ms=session_gap_ms)
    capture_keys = {path: f"capture_{extract_capture_condition(path)}" for path in paths}
    split_keys = {path: f"split_{split_origin_label(path)}" for path in paths}

    source_to_keys = {
        "session": session_keys,
        "capture_condition": capture_keys,
        "split_origin": split_keys,
    }

    selected_source = domain_label_source
    selected_keys = source_to_keys[selected_source]
    unique_count = len(set(selected_keys.values()))
    if selected_source == "session" and unique_count > max(32, len(paths) // 3):
        selected_source = "capture_condition"
        selected_keys = capture_keys
        unique_count = len(set(selected_keys.values()))
        if unique_count <= 1:
            selected_source = "split_origin"
            selected_keys = split_keys

    return [selected_keys[path] for path in paths], selected_source


class DomainLabeledDataset(Dataset):
    def __init__(self, base_dataset: Dataset, domain_labels: list[int]) -> None:
        if len(base_dataset) != len(domain_labels):
            raise ValueError("Domain label count must match dataset length.")
        self.base_dataset = base_dataset
        self.domain_labels = domain_labels

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        base_item = self.base_dataset[index]
        if len(base_item) == 2:
            image, label = base_item
            return image, label, self.domain_labels[index]
        image, label, landmarks = base_item
        return image, label, landmarks, self.domain_labels[index]


class LandmarkMetadataDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        metadata_root: Path | None = None,
        image_size: int = 112,
    ) -> None:
        self.base_dataset = base_dataset
        self.metadata_root = metadata_root
        self.image_size = float(image_size)
        self.paths = _dataset_paths(base_dataset)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _metadata_path_for(self, image_path: Path) -> Path | None:
        if self.metadata_root is not None:
            try:
                relative = image_path.relative_to(Path(image_path.anchor))
                candidate = self.metadata_root / relative.parent / f"{image_path.stem}.json"
            except ValueError:
                candidate = self.metadata_root / image_path.parent.name / f"{image_path.stem}.json"
            if candidate.exists():
                return candidate
        candidate = image_path.with_suffix(".json")
        return candidate if candidate.exists() else None

    def _load_landmarks(self, image_path: Path) -> torch.Tensor:
        metadata_path = self._metadata_path_for(image_path)
        if metadata_path is None:
            return torch.zeros((5, 2), dtype=torch.float32)

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_landmarks = (
            payload.get("landmarks")
            or payload.get("kps")
            or payload.get("keypoints")
            or payload.get("five_points")
        )
        if raw_landmarks is None:
            return torch.zeros((5, 2), dtype=torch.float32)
        landmarks = torch.tensor(raw_landmarks, dtype=torch.float32).view(5, 2)
        if torch.max(torch.abs(landmarks)) > 1.5:
            landmarks = landmarks / self.image_size
        return torch.clamp(landmarks, 0.0, 1.0)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        landmarks = self._load_landmarks(self.paths[index])
        return image, label, landmarks


def build_datasets(
    dataset_root: Path,
    val_split: float,
    seed: int,
    max_classes: int | None = None,
    max_samples_per_class: int | None = None,
    landmark_metadata_root: Path | None = None,
    motion_blur_prob: float = 0.0,
    motion_blur_kernel_size: int = 7,
) -> tuple[Dataset, Dataset, list[str], dict[str, Any]]:
    split_dirs = resolve_dataset_split_dirs(dataset_root)
    dataset_structure = "structured" if "train" in split_dirs else "flat"

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            DirectionalMotionBlur(probability=motion_blur_prob, kernel_size=motion_blur_kernel_size),
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
        if max_classes is not None:
            train_class_names = train_class_names[:max_classes]
            train_dataset_full = HierarchicalImageFolder(
                root=train_root,
                transform=train_transform,
                class_names=train_class_names,
            )
        train_dataset_full = maybe_limit_hierarchical_dataset(
            train_dataset_full,
            max_samples_per_class=max_samples_per_class,
        )
        val_dataset_full = HierarchicalImageFolder(
            root=val_root,
            transform=val_transform,
            class_names=train_class_names if val_root != train_root else None,
        )
        if val_root == train_root and max_classes is not None:
            val_dataset_full = HierarchicalImageFolder(
                root=val_root,
                transform=val_transform,
                class_names=train_class_names,
            )
        val_dataset_full = maybe_limit_hierarchical_dataset(
            val_dataset_full,
            max_samples_per_class=max_samples_per_class,
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
        class_names = list(train_dataset_full.classes)
        if max_classes is not None:
            class_names = class_names[:max_classes]
            train_dataset_full = HierarchicalImageFolder(
                root=dataset_root,
                transform=train_transform,
                class_names=class_names,
            )
        train_dataset_full = maybe_limit_hierarchical_dataset(
            train_dataset_full,
            max_samples_per_class=max_samples_per_class,
        )
        val_dataset_full = HierarchicalImageFolder(
            root=dataset_root,
            transform=val_transform,
            class_names=class_names if max_classes is not None else None,
        )
        val_dataset_full = maybe_limit_hierarchical_dataset(
            val_dataset_full,
            max_samples_per_class=max_samples_per_class,
        )
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

    if landmark_metadata_root is not None:
        train_dataset = LandmarkMetadataDataset(train_dataset, metadata_root=landmark_metadata_root)
        val_dataset = LandmarkMetadataDataset(val_dataset, metadata_root=landmark_metadata_root)

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
        "max_classes": max_classes,
        "max_samples_per_class": max_samples_per_class,
        "landmark_metadata_root": None if landmark_metadata_root is None else str(landmark_metadata_root),
        "motion_blur_prob": motion_blur_prob,
        "motion_blur_kernel_size": motion_blur_kernel_size,
    }
    return train_dataset, val_dataset, class_names, diagnostics


def format_optional_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def checkpoint_payload(
    model: torch.nn.Module,
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
        "kd_mode": args.kd_mode,
        "dataset_root": str(args.dataset_root),
        "seed": args.seed,
        "run_diagnostics": run_diagnostics,
    }
    payload.update(model.get_config().to_metadata())
    return payload


def load_teacher_center_cache(
    center_path: Path,
    class_names: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not center_path.exists():
        raise FileNotFoundError(f"Teacher centers cache not found: {center_path}")
    payload = torch.load(center_path, map_location="cpu")
    cache_class_names = list(payload["class_names"])
    if cache_class_names != class_names:
        raise ValueError(
            "Teacher center class mapping mismatch. "
            f"dataset_classes={class_names[:5]} cache_classes={cache_class_names[:5]}"
        )
    centers = payload["centers"].to(device)
    if centers.ndim != 2:
        raise ValueError(f"Teacher centers tensor must be 2D, got shape={tuple(centers.shape)}")
    diagnostics = {
        "path": str(center_path),
        "class_count": len(cache_class_names),
        "embedding_dim": int(payload["embedding_dim"]),
        "teacher_weights": payload.get("teacher_weights"),
    }
    return centers, diagnostics


def build_model_from_checkpoint(weight_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(weight_path, map_location="cpu")
    model, _ = build_model_from_metadata(checkpoint if isinstance(checkpoint, dict) else None)
    state_dict = clean_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint mismatch: missing={list(missing[:5])} unexpected={list(unexpected[:5])}"
        )
    return model


def build_teacher_model(
    *,
    teacher_backbone: str,
    teacher_weights: Path,
    embedding_dim: int,
    teacher_pretrained_imagenet: bool,
) -> torch.nn.Module:
    if teacher_backbone == "edgeface_ta":
        return build_model_from_checkpoint(teacher_weights)
    if teacher_backbone == "ir101_adaface":
        return IResNet101AdaFaceTeacher(embedding_size=embedding_dim)
    return ResNet101Teacher(
        embedding_size=embedding_dim,
        pretrained=teacher_pretrained_imagenet,
    )


def resolve_kd_enabled(args: argparse.Namespace) -> bool:
    if args.kd_mode == "none":
        return False
    if args.kd_mode == "embedding_mse":
        return args.kd_alpha > 0
    return True


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device()
    print(f"🚀 Đang huấn luyện trên thiết bị: {device}")

    if args.kd_alpha < 0:
        raise ValueError("kd_alpha must be >= 0.")
    if args.adadistill_weight < 0:
        raise ValueError("adadistill_weight must be >= 0.")
    if args.max_missing_ratio < 0 or args.max_unexpected_ratio < 0:
        raise ValueError("Bootstrap mismatch ratios must be >= 0.")
    if args.max_train_batches_per_epoch is not None and args.max_train_batches_per_epoch <= 0:
        raise ValueError("max_train_batches_per_epoch must be > 0 when provided.")
    if args.max_val_batches is not None and args.max_val_batches <= 0:
        raise ValueError("max_val_batches must be > 0 when provided.")
    if not (0.0 <= args.motion_blur_prob <= 1.0):
        raise ValueError("motion_blur_prob must be in [0, 1].")
    if args.motion_blur_kernel_size <= 0:
        raise ValueError("motion_blur_kernel_size must be > 0.")
    if args.adaface_h <= 0:
        raise ValueError("adaface_h must be > 0.")
    if not (0.0 < args.partial_fc_sample_rate <= 1.0):
        raise ValueError("partial_fc_sample_rate must be in (0, 1].")
    if args.partial_fc_min_negatives < 0:
        raise ValueError("partial_fc_min_negatives must be >= 0.")
    if args.kd_mode == "adadistill":
        if args.teacher_pretrained_imagenet:
            raise ValueError("AdaDistill faithful branch cannot use --teacher-pretrained-imagenet.")
        if args.teacher_centers_path is None:
            raise ValueError("kd-mode=adadistill requires --teacher-centers-path.")
        if args.teacher_backbone == "resnet101_imagenet":
            raise ValueError("kd-mode=adadistill requires teacher centers from ir101_adaface or edgeface_ta.")
    if (
        args.kd_mode == "embedding_mse"
        and args.kd_alpha > 0
        and args.skip_teacher_bootstrap
        and not args.teacher_pretrained_imagenet
        and args.teacher_backbone != "edgeface_ta"
    ):
        raise ValueError(
            "KD requires a compatible teacher. Use --teacher-pretrained-imagenet or do not skip teacher bootstrap."
        )
    if args.domain_loss_weight < 0:
        raise ValueError("domain_loss_weight must be >= 0.")
    if args.classifier_mode == "partial_fc" and args.kd_mode != "none":
        print("ℹ️ Partial FC is enabled together with KD. This is supported but intended mainly for AdaFace supervision.")

    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, class_names, dataset_diagnostics = build_datasets(
        dataset_root=args.dataset_root,
        val_split=args.val_split,
        seed=args.seed,
        max_classes=args.max_classes,
        max_samples_per_class=args.max_samples_per_class,
        landmark_metadata_root=args.landmark_metadata_root,
        motion_blur_prob=args.motion_blur_prob,
        motion_blur_kernel_size=args.motion_blur_kernel_size,
    )
    active_domain_source = None
    domain_key_to_index: dict[str, int] | None = None
    domain_adversarial_enabled = args.domain_adversarial
    if args.domain_adversarial:
        train_domain_keys, active_domain_source = resolve_domain_keys(
            train_dataset,
            dataset_root=args.dataset_root,
            domain_label_source=args.domain_label_source,
            session_gap_ms=args.domain_session_gap_ms,
        )
        ordered_domain_keys = sorted(set(train_domain_keys))
        if len(ordered_domain_keys) < 2:
            print("ℹ️ Domain adversarial disabled: fewer than 2 distinct domains after label resolution.")
            domain_adversarial_enabled = False
        else:
            domain_key_to_index = {domain_key: index for index, domain_key in enumerate(ordered_domain_keys)}
            train_domain_labels = [domain_key_to_index[key] for key in train_domain_keys]
            train_dataset = DomainLabeledDataset(train_dataset, train_domain_labels)

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

    student_model = build_model(
        backbone_name=args.backbone,
        embedding_dim=args.embedding_dim,
        width_preset=args.width_preset,
        stage_channels=args.stage_channels,
        rank_ratio=args.rank_ratio,
        attention_heads=args.attention_heads,
        attention_depth=args.attention_depth,
        kprpe_hidden_dim=args.kprpe_hidden_dim,
    )
    student_param_count = sum(parameter.numel() for parameter in student_model.parameters())

    kd_enabled = resolve_kd_enabled(args)
    active_classifier_mode = args.classifier_mode
    if args.classifier_mode == "partial_fc" and len(class_names) <= max(args.partial_fc_min_negatives, args.batch_size * 4):
        active_classifier_mode = "full"
    teacher_mode = "disabled"
    teacher_model: torch.nn.Module | None = None
    teacher_centers: torch.Tensor | None = None
    domain_grl: GradientReversalLayer | None = None
    domain_discriminator: DomainDiscriminator | None = None
    bootstrap_reports: dict[str, Any] = {
        "student": {
            "enabled": not args.skip_student_bootstrap,
            "path": str(args.student_weights),
        },
        "teacher": {
            "enabled": kd_enabled and args.kd_mode == "embedding_mse" and not args.skip_teacher_bootstrap and not args.teacher_pretrained_imagenet,
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

    if kd_enabled and args.kd_mode == "embedding_mse":
        teacher_model = build_teacher_model(
            teacher_backbone=args.teacher_backbone,
            teacher_weights=args.teacher_weights,
            embedding_dim=args.embedding_dim,
            teacher_pretrained_imagenet=args.teacher_pretrained_imagenet,
        )
        if args.teacher_pretrained_imagenet:
            teacher_mode = "imagenet_pretrained"
            bootstrap_reports["teacher"] = {
                "enabled": False,
                "reason": "teacher_pretrained_imagenet",
            }
            print("ℹ️ Teacher mode: torchvision ImageNet pretrained.")
        elif args.teacher_backbone == "edgeface_ta":
            teacher_mode = "edgeface_ta_checkpoint"
            bootstrap_reports["teacher"] = {
                "enabled": False,
                "reason": "edgeface_ta_checkpoint_loaded_in_builder",
                "path": str(args.teacher_weights),
            }
        else:
            teacher_mode = f"{args.teacher_backbone}_checkpoint_bootstrap"
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
    elif kd_enabled and args.kd_mode == "adadistill":
        teacher_mode = f"{args.teacher_backbone}_global_centers"
        teacher_centers, center_report = load_teacher_center_cache(args.teacher_centers_path, class_names, device)
        bootstrap_reports["teacher"] = {
            "enabled": False,
            "reason": "teacher_centers_cache",
            **center_report,
        }
    else:
        reason = "kd_disabled" if args.kd_mode != "none" else "kd_mode_none"
        print("ℹ️ KD: disabled.")
        bootstrap_reports["teacher"] = {"enabled": False, "reason": reason}

    student_model = student_model.to(device)
    if domain_adversarial_enabled:
        assert domain_key_to_index is not None and active_domain_source is not None
        domain_grl = GradientReversalLayer(lambda_=1.0).to(device)
        domain_discriminator = DomainDiscriminator(
            input_dim=args.embedding_dim,
            num_domains=len(domain_key_to_index),
        ).to(device)

    run_diagnostics: dict[str, Any] = {
        "dataset": dataset_diagnostics,
        "student_param_count": student_param_count,
        "student_bootstrap": bootstrap_reports["student"],
        "teacher_bootstrap": bootstrap_reports["teacher"],
        "kd_enabled": kd_enabled,
        "kd_mode": args.kd_mode,
        "kd_alpha": args.kd_alpha,
        "backbone": args.backbone,
        "teacher_backbone": args.teacher_backbone,
        "teacher_mode": teacher_mode,
        "teacher_centers_path": str(args.teacher_centers_path) if args.teacher_centers_path is not None else None,
        "adaface_margin": args.adaface_margin,
        "adaface_scale": args.adaface_scale,
        "adaface_h": args.adaface_h,
        "classifier_mode": active_classifier_mode,
        "partial_fc_sample_rate": args.partial_fc_sample_rate,
        "partial_fc_min_negatives": args.partial_fc_min_negatives,
        "adadistill_weight": args.adadistill_weight,
        "teacher_logit_scale": args.teacher_logit_scale,
        "device": str(device),
        "output_prefix": args.output_prefix,
        "max_train_batches_per_epoch": args.max_train_batches_per_epoch,
        "max_val_batches": args.max_val_batches,
        "max_classes": args.max_classes,
        "max_samples_per_class": args.max_samples_per_class,
        "domain_adversarial": domain_adversarial_enabled,
        "domain_loss_weight": args.domain_loss_weight,
        "domain_label_source": active_domain_source if active_domain_source is not None else args.domain_label_source,
        "domain_count": len(domain_key_to_index) if domain_key_to_index is not None else 0,
        "domain_session_gap_ms": args.domain_session_gap_ms,
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
    print(f"🧭 KD enabled: {kd_enabled} | KD mode: {args.kd_mode} | Teacher mode: {teacher_mode}")
    if domain_adversarial_enabled:
        print(
            f"🧭 Domain adversarial enabled | source: {run_diagnostics['domain_label_source']} | "
            f"domains: {run_diagnostics['domain_count']} | weight: {args.domain_loss_weight}"
        )
    print(
        f"🧭 Partial epoch: train_batches="
        f"{args.max_train_batches_per_epoch if args.max_train_batches_per_epoch is not None else 'full'} | "
        f"val_batches={args.max_val_batches if args.max_val_batches is not None else 'full'}"
    )

    adaface_criterion = AdaFaceLoss(
        embedding_size=args.embedding_dim,
        num_classes=len(class_names),
        margin=args.adaface_margin,
        scale=args.adaface_scale,
        h=args.adaface_h,
    ).to(device)
    partial_fc_generator = torch.Generator(device="cpu")
    partial_fc_generator.manual_seed(args.partial_fc_seed)
    kd_criterion = EmbeddingKDLoss(alpha=args.kd_alpha).to(device) if kd_enabled and args.kd_mode == "embedding_mse" else None
    adadistill_criterion = (
        AdaDistillLoss(weight=args.adadistill_weight, scale=args.teacher_logit_scale).to(device)
        if kd_enabled and args.kd_mode == "adadistill"
        else None
    )
    optimizer_params = [
        {"params": student_model.parameters(), "weight_decay": 5e-4},
        {"params": adaface_criterion.parameters(), "weight_decay": 5e-4},
    ]
    if domain_discriminator is not None:
        optimizer_params.append({"params": domain_discriminator.parameters(), "weight_decay": 5e-4})
    optimizer = optim.AdamW(optimizer_params, lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    scaler = amp.GradScaler(device_type, enabled=(device_type == "cuda"))

    history: list[dict[str, Any]] = []
    best_val_accuracy = float("-inf")
    best_checkpoint_path = args.checkpoints_dir / f"{args.output_prefix}_best.pth"

    for epoch in range(args.epochs):
        student_model.train()
        if domain_discriminator is not None:
            domain_discriminator.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        train_cosine_sum = 0.0
        train_cosine_batches = 0
        domain_loss_sum = 0.0
        domain_correct = 0
        domain_total = 0
        adadistill_p_sum = 0.0
        adadistill_gt_conf_sum = 0.0
        adadistill_easy_sum = 0.0
        adadistill_hard_sum = 0.0
        adadistill_batches = 0
        processed_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if (
                args.max_train_batches_per_epoch is not None
                and batch_idx >= args.max_train_batches_per_epoch
            ):
                break
            if domain_adversarial_enabled:
                if len(batch) == 4:
                    images, labels, landmarks, domain_labels = batch
                else:
                    images, labels, domain_labels = batch
                    landmarks = None
                domain_labels = domain_labels.to(device)
            else:
                if len(batch) == 3:
                    images, labels, landmarks = batch
                else:
                    images, labels = batch
                    landmarks = None
                domain_labels = None
            images = images.to(device)
            labels = labels.to(device)
            if landmarks is not None:
                landmarks = landmarks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                student_embeddings, student_norms = student_model(images, landmarks=landmarks)
                loss_adaface = adaface_criterion(
                    student_embeddings,
                    student_norms,
                    labels,
                    classifier_mode=active_classifier_mode,
                    partial_fc_sample_rate=args.partial_fc_sample_rate,
                    partial_fc_min_negatives=args.partial_fc_min_negatives,
                    partial_fc_generator=partial_fc_generator,
                )

                teacher_embeddings = None
                adadistill_stats: AdaDistillStats | None = None
                if kd_enabled and args.kd_mode == "embedding_mse":
                    assert teacher_model is not None and kd_criterion is not None
                    with torch.no_grad():
                        teacher_inputs = prepare_teacher_inputs(images, args.teacher_backbone)
                        teacher_embeddings, _ = teacher_model(teacher_inputs)
                    loss_kd = kd_criterion(student_embeddings, teacher_embeddings)
                elif kd_enabled and args.kd_mode == "adadistill":
                    assert adadistill_criterion is not None and teacher_centers is not None
                    loss_kd, adadistill_stats = adadistill_criterion(
                        student_embeddings=student_embeddings,
                        student_classifier_weights=adaface_criterion.weights,
                        teacher_centers=teacher_centers,
                        labels=labels,
                    )
                else:
                    loss_kd = torch.zeros((), device=device)

                if domain_adversarial_enabled:
                    assert domain_grl is not None and domain_discriminator is not None and domain_labels is not None
                    domain_logits = domain_discriminator(domain_grl(student_embeddings))
                    loss_domain = F.cross_entropy(domain_logits, domain_labels)
                else:
                    domain_logits = None
                    loss_domain = torch.zeros((), device=device)

                loss = loss_adaface + loss_kd + (args.domain_loss_weight * loss_domain)

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
            if domain_logits is not None and domain_labels is not None:
                domain_loss_sum += loss_domain.item()
                domain_predictions = domain_logits.argmax(dim=1)
                domain_correct += (domain_predictions == domain_labels).sum().item()
                domain_total += domain_labels.size(0)
            if adadistill_stats is not None:
                adadistill_p_sum += adadistill_stats.adaptive_p
                adadistill_gt_conf_sum += adadistill_stats.gt_confidence
                adadistill_easy_sum += adadistill_stats.loss_easy
                adadistill_hard_sum += adadistill_stats.loss_hard
                adadistill_batches += 1
            processed_train_batches += 1

            if batch_idx % 10 == 0:
                kd_log = "disabled" if not kd_enabled else f"{loss_kd.item():.4f}"
                train_batch_total = (
                    args.max_train_batches_per_epoch
                    if args.max_train_batches_per_epoch is not None
                    else len(train_loader)
                )
                extra_kd_log = ""
                if adadistill_stats is not None:
                    extra_kd_log = (
                        f", p: {adadistill_stats.adaptive_p:.4f}, "
                        f"gt_conf: {adadistill_stats.gt_confidence:.4f}, "
                        f"L_easy: {adadistill_stats.loss_easy:.4f}, "
                        f"L_hard: {adadistill_stats.loss_hard:.4f}"
                    )
                if domain_logits is not None:
                    extra_kd_log += f", Domain: {loss_domain.item():.4f}"
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] | Batch [{batch_idx}/{train_batch_total}] | "
                    f"Loss: {loss.item():.4f} (AdaFace: {loss_adaface.item():.4f}, KD: {kd_log}{extra_kd_log})"
                )

        train_accuracy = 100.0 * train_correct / max(1, train_total)
        train_cosine = (
            train_cosine_sum / max(1, train_cosine_batches) if kd_enabled and train_cosine_batches > 0 else None
        )
        train_adadistill_p = adadistill_p_sum / max(1, adadistill_batches) if adadistill_batches > 0 else None
        train_adadistill_gt_conf = (
            adadistill_gt_conf_sum / max(1, adadistill_batches) if adadistill_batches > 0 else None
        )
        train_adadistill_easy = adadistill_easy_sum / max(1, adadistill_batches) if adadistill_batches > 0 else None
        train_adadistill_hard = adadistill_hard_sum / max(1, adadistill_batches) if adadistill_batches > 0 else None
        train_domain_accuracy = 100.0 * domain_correct / max(1, domain_total) if domain_adversarial_enabled else None
        train_domain_loss = domain_loss_sum / max(1, processed_train_batches) if domain_adversarial_enabled else None

        student_model.eval()
        val_correct = 0
        val_total = 0
        val_cosine_sum = 0.0
        val_cosine_batches = 0
        processed_val_batches = 0
        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_loader):
                if args.max_val_batches is not None and val_batch_idx >= args.max_val_batches:
                    break
                if len(batch) == 3:
                    images, labels, landmarks = batch
                    landmarks = landmarks.to(device)
                else:
                    images, labels = batch
                    landmarks = None
                images = images.to(device)
                labels = labels.to(device)
                student_embeddings, student_norms = student_model(images, landmarks=landmarks)
                logits = compute_classification_logits(student_embeddings, adaface_criterion.weights)
                predictions = logits.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                processed_val_batches += 1

                if kd_enabled and args.kd_mode == "embedding_mse":
                    assert teacher_model is not None
                    teacher_inputs = prepare_teacher_inputs(images, args.teacher_backbone)
                    teacher_embeddings, _ = teacher_model(teacher_inputs)
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
                "kd_mode": args.kd_mode,
                "train_adadistill_p": train_adadistill_p,
                "train_adadistill_gt_conf": train_adadistill_gt_conf,
                "train_adadistill_easy": train_adadistill_easy,
                "train_adadistill_hard": train_adadistill_hard,
                "train_domain_loss": train_domain_loss,
                "train_domain_accuracy": train_domain_accuracy,
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
        if train_adadistill_p is not None:
            print(
                f"   ↳ AdaDistill | p: {train_adadistill_p:.4f} | gt_conf: {train_adadistill_gt_conf:.4f} | "
                f"L_easy: {train_adadistill_easy:.4f} | L_hard: {train_adadistill_hard:.4f}"
            )
        if train_domain_loss is not None:
            print(
                f"   ↳ Domain | loss: {train_domain_loss:.4f} | train_acc: {train_domain_accuracy:.2f}%"
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
