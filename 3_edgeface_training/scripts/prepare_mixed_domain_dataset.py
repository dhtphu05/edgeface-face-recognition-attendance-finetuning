from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.dataset import discover_class_dir_map, resolve_dataset_split_dirs

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a canonical mixed-domain dataset by merging multiple sources.")
    parser.add_argument("--source", action="append", required=True, help="Source in the form name=/abs/path")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copy-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def parse_source(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Invalid --source value: {raw}. Expected name=/path")
    name, raw_path = raw.split("=", 1)
    source_root = Path(raw_path).expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    return name.strip(), source_root


def link_or_copy(src: Path, dst: Path, copy_mode: str) -> None:
    if copy_mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def main() -> None:
    args = parse_args()
    sources = [parse_source(raw) for raw in args.source]
    if args.clear_output and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "output_root": str(args.output_root),
        "copy_mode": args.copy_mode,
        "sources": [],
    }

    for source_name, source_root in sources:
        split_dirs = resolve_dataset_split_dirs(source_root)
        source_entry = {"name": source_name, "root": str(source_root), "splits": {}}
        if "all" in split_dirs:
            iterable_splits = {"train": split_dirs["all"]}
        else:
            iterable_splits = dict(split_dirs)

        for split_name, split_root in iterable_splits.items():
            output_split_root = args.output_root / split_name
            output_split_root.mkdir(parents=True, exist_ok=True)
            class_dir_map = discover_class_dir_map(split_root)
            created_classes = 0
            linked_images = 0

            for class_name, class_dir in class_dir_map.items():
                mixed_class_name = f"{source_name}__{class_name}"
                output_class_dir = output_split_root / mixed_class_name
                output_class_dir.mkdir(parents=True, exist_ok=True)
                created_classes += 1
                for image_path in sorted(class_dir.iterdir()):
                    if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    output_path = output_class_dir / image_path.name
                    if output_path.exists():
                        continue
                    link_or_copy(image_path, output_path, args.copy_mode)
                    linked_images += 1

                    metadata_path = image_path.with_suffix(".json")
                    if metadata_path.exists():
                        output_metadata_path = output_class_dir / metadata_path.name
                        if not output_metadata_path.exists():
                            link_or_copy(metadata_path, output_metadata_path, args.copy_mode)

            source_entry["splits"][split_name] = {
                "class_count": created_classes,
                "image_count": linked_images,
            }

        manifest["sources"].append(source_entry)

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved mixed-domain manifest to {manifest_path}")


if __name__ == "__main__":
    main()
