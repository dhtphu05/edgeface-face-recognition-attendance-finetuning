from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pgm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonicalize a ChokePoint-style portal dataset into train/val/test identity folders."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Root like external_datasets/P1E")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copy-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument(
        "--split-strategy",
        choices=["all_train", "by_session"],
        default="by_session",
        help="all_train puts everything under train; by_session maps S1/S2->train, S3->val, S4->test by default.",
    )
    parser.add_argument(
        "--session-split",
        action="append",
        default=[],
        help="Override split mapping with values like S1=train. Can be repeated.",
    )
    parser.add_argument(
        "--prefix-identities",
        default="chokepoint",
        help="Prefix for canonical class names. Example: chokepoint__0001",
    )
    return parser.parse_args()


def parse_session_split(overrides: list[str]) -> dict[str, str]:
    mapping = {"S1": "train", "S2": "train", "S3": "val", "S4": "test"}
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"Invalid --session-split value: {raw}. Expected S1=train")
        session, split_name = raw.split("=", 1)
        session = session.strip().upper()
        split_name = split_name.strip().lower()
        if split_name not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split name in --session-split: {raw}")
        mapping[session] = split_name
    return mapping


def extract_session_camera(sequence_name: str) -> tuple[str, str]:
    parts = sequence_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected sequence name format: {sequence_name}")
    session = parts[1].upper()
    camera = parts[2].upper()
    return session, camera


def link_or_copy(src: Path, dst: Path, copy_mode: str) -> None:
    if copy_mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def build_metadata(
    *,
    image_path: Path,
    sequence_name: str,
    session: str,
    camera: str,
    identity_name: str,
    original_ext: str,
) -> dict[str, object]:
    return {
        "source_dataset": "chokepoint",
        "sequence": sequence_name,
        "session": session,
        "camera": camera,
        "identity": identity_name,
        "original_path": str(image_path),
        "original_extension": original_ext,
        "image_size": [96, 96],
        "recommended_resize": [112, 112],
    }


def discover_sequences(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.iterdir() if path.is_dir())


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    if args.clear_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    session_split = parse_session_split(args.session_split)
    sequence_paths = discover_sequences(input_root)
    if not sequence_paths:
        raise ValueError(f"No sequence folders found under: {input_root}")

    manifest: dict[str, object] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "copy_mode": args.copy_mode,
        "split_strategy": args.split_strategy,
        "session_split": session_split if args.split_strategy == "by_session" else {},
        "sequences": {},
        "summary": {},
    }

    split_identity_counts: dict[str, set[str]] = defaultdict(set)
    split_image_counts: Counter[str] = Counter()

    for sequence_path in sequence_paths:
        sequence_name = sequence_path.name
        session, camera = extract_session_camera(sequence_name)
        split_name = "train" if args.split_strategy == "all_train" else session_split.get(session, "train")
        split_root = output_root / split_name
        split_root.mkdir(parents=True, exist_ok=True)

        sequence_summary = {
            "session": session,
            "camera": camera,
            "split": split_name,
            "identity_count": 0,
            "image_count": 0,
        }

        identity_dirs = sorted(path for path in sequence_path.iterdir() if path.is_dir())
        for identity_dir in identity_dirs:
            identity_name = identity_dir.name
            canonical_identity = f"{args.prefix_identities}__{identity_name}"
            output_identity_dir = split_root / canonical_identity
            output_identity_dir.mkdir(parents=True, exist_ok=True)
            split_identity_counts[split_name].add(canonical_identity)
            sequence_summary["identity_count"] += 1

            image_paths = sorted(
                path
                for path in identity_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            for image_path in image_paths:
                filename = f"{sequence_name}__{image_path.name}"
                output_image_path = output_identity_dir / filename
                if not output_image_path.exists():
                    link_or_copy(image_path, output_image_path, args.copy_mode)

                metadata = build_metadata(
                    image_path=image_path,
                    sequence_name=sequence_name,
                    session=session,
                    camera=camera,
                    identity_name=identity_name,
                    original_ext=image_path.suffix.lower(),
                )
                output_metadata_path = output_identity_dir / f"{output_image_path.stem}.json"
                output_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

                sequence_summary["image_count"] += 1
                split_image_counts[split_name] += 1

        manifest["sequences"][sequence_name] = sequence_summary

    manifest["summary"] = {
        split_name: {
            "identity_count": len(identity_names),
            "image_count": split_image_counts[split_name],
        }
        for split_name, identity_names in sorted(split_identity_counts.items())
    }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved ChokePoint manifest to {manifest_path}")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
