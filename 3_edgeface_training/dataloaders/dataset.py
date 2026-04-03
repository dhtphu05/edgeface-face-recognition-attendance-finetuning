from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int


class FaceFolderDataset(Dataset):
    def __init__(self, root_dir: str | Path, image_size: int = 112) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.class_names = sorted(
            [path.name for path in self.root_dir.iterdir() if path.is_dir()]
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples = self._discover_samples()

        if not self.samples:
            raise ValueError(f"No images found in dataset directory: {self.root_dir}")

    def _discover_samples(self) -> list[Sample]:
        samples: list[Sample] = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS and image_path.is_file():
                    samples.append(Sample(image_path=image_path, label=self.class_to_idx[class_name]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = (image_np - 0.5) / 0.5
        image_np = np.transpose(image_np, (2, 0, 1))
        return torch.from_numpy(image_np), sample.label


def build_dataloaders(
    dataset_root: str | Path,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str]]:
    dataset = FaceFolderDataset(dataset_root)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested validation split.")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, dataset.class_names
