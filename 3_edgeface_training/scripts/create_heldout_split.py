from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a held-out dataset split by session/time groups.")
    parser.add_argument("--source-root", type=Path, default=WORKSPACE_ROOT / "2_face_dataset")
    parser.add_argument("--output-root", type=Path, default=WORKSPACE_ROOT / "2_face_dataset_split")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--session-gap-ms",
        type=int,
        default=3000,
        help="New session boundary if adjacent timestamps differ by more than this gap.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["symlink", "copy"],
        default="symlink",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the existing output-root before creating a new split.",
    )
    return parser.parse_args()


def extract_timestamp(path: Path) -> int | None:
    digits = [token for token in path.stem.split("_") if token.isdigit()]
    if not digits:
        return None
    return int(max(digits, key=len))


def group_sessions(image_paths: list[Path], session_gap_ms: int) -> list[list[Path]]:
    ordered = sorted(image_paths, key=lambda path: (extract_timestamp(path) or 0, path.name))
    sessions: list[list[Path]] = []
    current: list[Path] = []
    last_ts: int | None = None

    for image_path in ordered:
        current_ts = extract_timestamp(image_path)
        if (
            current
            and current_ts is not None
            and last_ts is not None
            and current_ts - last_ts > session_gap_ms
        ):
            sessions.append(current)
            current = []
        current.append(image_path)
        last_ts = current_ts

    if current:
        sessions.append(current)
    return sessions


def target_image_counts(total_images: int, val_ratio: float, test_ratio: float) -> dict[str, int]:
    test_target = round(total_images * test_ratio)
    val_target = round(total_images * val_ratio)
    train_target = total_images - test_target - val_target
    return {"train": train_target, "val": val_target, "test": test_target}


def assign_sessions_balanced(sessions: list[list[Path]], val_ratio: float, test_ratio: float) -> dict[str, list[int]]:
    session_sizes = [(index, len(session)) for index, session in enumerate(sessions)]
    session_sizes.sort(key=lambda item: item[1], reverse=True)

    split_targets = target_image_counts(sum(size for _, size in session_sizes), val_ratio, test_ratio)
    split_assignments: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0}

    for session_index, session_size in session_sizes:
        candidates = []
        for split_name in ("train", "val", "test"):
            projected = split_counts[split_name] + session_size
            overflow = max(0, projected - split_targets[split_name])
            deficit = max(0, split_targets[split_name] - split_counts[split_name])
            candidates.append((overflow, -deficit, split_counts[split_name], split_name))
        _, _, _, chosen_split = min(candidates)
        split_assignments[chosen_split].append(session_index)
        split_counts[chosen_split] += session_size

    non_empty_splits = sum(1 for split in split_assignments.values() if split)
    if len(sessions) >= 3 and non_empty_splits < 3:
        rebalance_small_session_splits(sessions, split_assignments, split_counts)

    return split_assignments


def rebalance_small_session_splits(
    sessions: list[list[Path]],
    split_assignments: dict[str, list[int]],
    split_counts: dict[str, int],
) -> None:
    for required_split in ("train", "val", "test"):
        if split_assignments[required_split]:
            continue

        donor_split = max(
            (split for split in ("train", "val", "test") if split_assignments[split]),
            key=lambda split: split_counts[split],
        )
        donor_index = min(split_assignments[donor_split], key=lambda idx: len(sessions[idx]))
        split_assignments[donor_split].remove(donor_index)
        split_counts[donor_split] -= len(sessions[donor_index])
        split_assignments[required_split].append(donor_index)
        split_counts[required_split] += len(sessions[donor_index])


def materialize_file(source: Path, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if copy_mode == "copy":
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def maybe_clear_output(output_root: Path, clear_output: bool) -> None:
    if clear_output and output_root.exists():
        shutil.rmtree(output_root)


def main() -> None:
    args = parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to less than 1.")

    class_dirs = sorted([path for path in args.source_root.iterdir() if path.is_dir()])
    if not class_dirs:
        raise ValueError(f"No identity folders found under: {args.source_root}")

    maybe_clear_output(args.output_root, args.clear_output)
    split_counts = defaultdict(int)
    per_class_counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        (args.output_root / split_name).mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        image_paths = sorted([path for path in class_dir.iterdir() if path.is_file()])
        if len(image_paths) < 3:
            print(f"⚠️ Bỏ qua {class_dir.name}: cần ít nhất 3 ảnh để tạo held-out split.")
            continue

        sessions = group_sessions(image_paths, session_gap_ms=args.session_gap_ms)
        split_indices = assign_sessions_balanced(sessions, args.val_ratio, args.test_ratio)
        class_counts = {"train": 0, "val": 0, "test": 0}

        for split_name, indices in split_indices.items():
            for index in indices:
                for image_path in sessions[index]:
                    destination = args.output_root / split_name / class_dir.name / image_path.name
                    materialize_file(image_path, destination, args.copy_mode)
                    split_counts[split_name] += 1
                    class_counts[split_name] += 1

        per_class_counts[class_dir.name] = class_counts
        session_sizes = [len(sessions[index]) for index in range(len(sessions))]
        print(
            f"✅ {class_dir.name}: sessions={len(sessions)} sizes={session_sizes} "
            f"train_images={class_counts['train']} val_images={class_counts['val']} test_images={class_counts['test']}"
        )

    print(
        f"Hoàn tất held-out split tại {args.output_root} | "
        f"train_images={split_counts['train']} val_images={split_counts['val']} test_images={split_counts['test']}"
    )
    print(f"Per-class split counts: {per_class_counts}")


if __name__ == "__main__":
    main()
