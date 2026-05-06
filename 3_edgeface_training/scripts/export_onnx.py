from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_factory import build_model_from_metadata


class RecognitionOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings, norms = self.model(images)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings, normalized_embeddings, norms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained recognition checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the .pth checkpoint to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output .onnx file.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export with a dynamic batch dimension.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run ONNX checker after export.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path) -> RecognitionOnnxWrapper:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    else:
        state_dict = checkpoint
        metadata = {}

    model, _ = build_model_from_metadata(metadata)
    model.load_state_dict(state_dict)
    model.eval()
    return RecognitionOnnxWrapper(model)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = load_model(args.checkpoint)
    dummy_input = torch.randn(1, 3, 112, 112, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    input_names = ["images"]
    output_names = ["embeddings", "normalized_embeddings", "norms"]
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch"},
            "embeddings": {0: "batch"},
            "normalized_embeddings": {0: "batch"},
            "norms": {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(args.output),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    if args.verify:
        exported_model = onnx.load(str(args.output))
        onnx.checker.check_model(exported_model)

    print(f"Exported ONNX model to: {args.output}")


if __name__ == "__main__":
    main()
