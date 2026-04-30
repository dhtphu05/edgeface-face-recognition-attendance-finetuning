from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.dataset import IMAGE_EXTENSIONS, discover_class_dir_map, resolve_dataset_split_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit face dataset quality before retraining.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--burst-gap-ms", type=int, default=400)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def extract_timestamp(path: Path) -> int | None:
    digits = [token for token in path.stem.split("_") if token.isdigit()]
    if not digits:
        return None
    return int(max(digits, key=len))


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    split_dirs = resolve_dataset_split_dirs(args.dataset_root)
    report: dict[str, object] = {"dataset_root": str(args.dataset_root), "splits": {}}

    for split_name, split_root in split_dirs.items():
        class_dir_map = discover_class_dir_map(split_root)
        if not class_dir_map:
            raise ValueError(f"No identity folders found under split {split_name}: {split_root}")

        split_report: dict[str, object] = {"classes": {}}
        class_counts: dict[str, int] = {}

        for class_name, class_dir in class_dir_map.items():
            image_paths = sorted(
                path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            class_counts[class_name] = len(image_paths)

            duplicate_hashes: dict[str, list[str]] = defaultdict(list)
            timestamps = []
            for image_path in image_paths:
                duplicate_hashes[sha1_file(image_path)].append(image_path.name)
                timestamp = extract_timestamp(image_path)
                if timestamp is not None:
                    timestamps.append((timestamp, image_path.name))

            exact_duplicates = [names for names in duplicate_hashes.values() if len(names) > 1]
            timestamps.sort()
            burst_pairs = []
            for previous, current in zip(timestamps, timestamps[1:]):
                delta = current[0] - previous[0]
                if delta <= args.burst_gap_ms:
                    burst_pairs.append(
                        {
                            "previous": previous[1],
                            "current": current[1],
                            "delta_ms": delta,
                        }
                    )

            split_report["classes"][class_name] = {
                "image_count": len(image_paths),
                "exact_duplicate_groups": exact_duplicates,
                "burst_candidates": burst_pairs[:50],
                "session_estimate": _estimate_sessions([ts for ts, _ in timestamps]),
            }

        counts = list(class_counts.values())
        median_count = sorted(counts)[len(counts) // 2]
        imbalance = {
            class_name: count
            for class_name, count in class_counts.items()
            if count < median_count * 0.5 or count > median_count * 2.0
        }
        split_report["summary"] = {
            "class_count": len(class_counts),
            "total_images": sum(class_counts.values()),
            "median_images_per_class": median_count,
            "imbalanced_classes": imbalance,
        }
        report["splits"][split_name] = split_report

    summary = {split_name: split_report["summary"] for split_name, split_report in report["splits"].items()}
    print(json.dumps(summary, indent=2))
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"📝 Đã lưu báo cáo audit tại: {args.report_json}")


def _estimate_sessions(timestamps: list[int], gap_ms: int = 3000) -> int:
    if not timestamps:
        return 1
    sessions = 1
    for previous, current in zip(timestamps, timestamps[1:]):
        if current - previous > gap_ms:
            sessions += 1
    return sessions


if __name__ == "__main__":
    main()
