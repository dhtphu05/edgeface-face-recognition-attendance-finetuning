from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a clean-core dataset by keeping files that start with their student_id."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--copy-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to materialize selected files into the clean-core dataset.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the existing output-root before creating the clean-core dataset.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap on the number of clean-core images kept per class.",
    )
    return parser.parse_args()


def materialize_file(source: Path, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if copy_mode == "copy":
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def main() -> None:
    args = parse_args()
    if not args.source_root.is_dir():
        raise FileNotFoundError(f"Source dataset directory not found: {args.source_root}")
    if args.max_per_class is not None and args.max_per_class <= 0:
        raise ValueError("max-per-class must be greater than 0 when provided.")

    if args.clear_output and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    total_selected = 0
    total_original = 0
    class_dirs = sorted(path for path in args.source_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found under: {args.source_root}")

    for class_dir in class_dirs:
        candidate_images = [
            path
            for path in sorted(class_dir.iterdir())
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.name.startswith(f"{class_dir.name}_")
        ]
        all_images = [
            path
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        total_original += len(all_images)

        if args.max_per_class is not None:
            candidate_images = candidate_images[: args.max_per_class]

        output_class_dir = args.output_root / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        for image_path in candidate_images:
            materialize_file(
                source=image_path,
                destination=output_class_dir / image_path.name,
                copy_mode=args.copy_mode,
            )

        total_selected += len(candidate_images)
        print(
            f"✅ {class_dir.name}: selected={len(candidate_images)} total_images={len(all_images)} "
            f"selection_ratio={(len(candidate_images) / max(1, len(all_images))):.2%}"
        )

    print(
        f"Created clean-core dataset at {args.output_root} | "
        f"selected_images={total_selected} total_source_images={total_original}"
    )


if __name__ == "__main__":
    main()
