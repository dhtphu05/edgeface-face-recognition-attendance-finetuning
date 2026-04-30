from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunSummary:
    metrics_path: Path
    label: str
    best_val_accuracy: float
    final_train_loss: float
    kd_alpha: float | None
    output_prefix: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the winner across public training runs using the FO v1 tie-break rule."
    )
    parser.add_argument(
        "metrics_paths",
        nargs="+",
        type=Path,
        help="One or more metrics JSON files emitted by train_phase3.py.",
    )
    parser.add_argument(
        "--close-threshold",
        type=float,
        default=0.2,
        help="If best val accuracy differs by at most this amount, compare final train loss next.",
    )
    return parser.parse_args()


def load_summary(metrics_path: Path) -> RunSummary:
    payload = json.loads(metrics_path.read_text())
    history: list[dict[str, Any]] = payload.get("history", [])
    if not history:
        raise ValueError(f"No history found in metrics file: {metrics_path}")

    last_epoch = history[-1]
    run_diagnostics = payload.get("run_diagnostics", {})
    kd_alpha = run_diagnostics.get("kd_alpha")
    output_prefix = run_diagnostics.get("output_prefix")

    label = output_prefix or metrics_path.stem.replace("_metrics", "")
    best_val_accuracy = float(payload["best_val_accuracy"])
    final_train_loss = float(last_epoch["train_loss"])

    return RunSummary(
        metrics_path=metrics_path,
        label=label,
        best_val_accuracy=best_val_accuracy,
        final_train_loss=final_train_loss,
        kd_alpha=float(kd_alpha) if kd_alpha is not None else None,
        output_prefix=output_prefix,
    )


def candidate_sort_key(run: RunSummary) -> tuple[float, float, float]:
    kd_alpha = run.kd_alpha if run.kd_alpha is not None else float("inf")
    return (-run.best_val_accuracy, run.final_train_loss, kd_alpha)


def select_winner(runs: list[RunSummary], close_threshold: float) -> RunSummary:
    if len(runs) == 1:
        return runs[0]

    runs_by_val = sorted(runs, key=lambda run: run.best_val_accuracy, reverse=True)
    top_val = runs_by_val[0].best_val_accuracy
    close_group = [
        run for run in runs_by_val if (top_val - run.best_val_accuracy) <= close_threshold
    ]
    return sorted(close_group, key=candidate_sort_key)[0]


def main() -> None:
    args = parse_args()
    runs = [load_summary(metrics_path.resolve()) for metrics_path in args.metrics_paths]
    winner = select_winner(runs, close_threshold=args.close_threshold)

    ranked = sorted(runs, key=candidate_sort_key)

    print("FO public run ranking:")
    for index, run in enumerate(ranked, start=1):
        kd_alpha = "n/a" if run.kd_alpha is None else f"{run.kd_alpha:g}"
        print(
            f"{index}. {run.label} | best_val={run.best_val_accuracy:.4f} | "
            f"final_train_loss={run.final_train_loss:.4f} | kd_alpha={kd_alpha} | "
            f"path={run.metrics_path}"
        )

    print("\nWinner:")
    print(
        f"{winner.label} | best_val={winner.best_val_accuracy:.4f} | "
        f"final_train_loss={winner.final_train_loss:.4f} | "
        f"kd_alpha={'n/a' if winner.kd_alpha is None else f'{winner.kd_alpha:g}'}"
    )


if __name__ == "__main__":
    main()
