from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_curve
from thop import clever_format, profile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.dataset import resolve_dataset_split_dirs
from models.edgeface_xxs import EdgeFaceXXS, build_edgeface_config_from_metadata
from scripts.prune_phase4 import apply_structured_pruning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a face recognition checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=WORKSPACE_ROOT / "2_face_dataset")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "phase5_final_model.pth",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--apply-pruning", action="store_true")
    parser.add_argument("--target-far", type=float, default=1e-3)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def measure_complexity_and_latency(model, device):
    model.eval()
    dummy_input = torch.randn(1, 3, 112, 112).to(device)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2
    flops_str, params_str = clever_format([flops, params], "%.2f")

    print("-" * 50)
    print("📊 ĐÁNH GIÁ ĐỘ PHỨC TẠP VÀ TỐC ĐỘ PHẦN CỨNG")
    print(f"Tổng số Tham số (Params): {params_str}")
    print(f"Tổng khối lượng tính toán (FLOPs): {flops_str}")

    print("⏳ Đang đo lường Latency (Giả lập suy luận 1000 lần)...")
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

        start_time = time.time()
        for _ in range(1000):
            _ = model(dummy_input)
        end_time = time.time()

    avg_latency = ((end_time - start_time) / 1000) * 1000
    fps = 1000 / avg_latency
    print(f"Độ trễ trung bình (Latency): {avg_latency:.2f} ms / khung hình")
    print(f"Tốc độ khung hình (FPS): {fps:.2f} FPS")
    print("-" * 50)
    return {
        "params": int(params),
        "flops": float(flops),
        "latency_ms": float(avg_latency),
        "fps": float(fps),
    }


def measure_biometrics(model, dataloader, device, target_far: float):
    model.eval()
    embeddings_list = []
    labels_list = []

    print("🔍 Đang trích xuất đặc trưng (Embeddings) cho toàn bộ tập Test...")
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.append(labels.numpy())

    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list)

    print("🧮 Đang tính toán ma trận Cosine Similarity...")
    cosine_dists = pdist(embeddings, metric="cosine")
    similarities = 1 - cosine_dists

    pair_labels = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pair_labels.append(1 if labels[i] == labels[j] else 0)
    pair_labels = np.array(pair_labels)

    print("📈 Đang phân tích đường cong ROC...")
    fpr, tpr, thresholds = roc_curve(pair_labels, similarities)
    frr = 1 - tpr
    far = fpr

    idx_target_far = np.abs(far - target_far).argmin()
    optimal_threshold = thresholds[idx_target_far]
    frr_at_target_far = frr[idx_target_far]
    predictions = (similarities >= optimal_threshold).astype(int)
    accuracy = np.mean(predictions == pair_labels) * 100

    print("🧬 ĐÁNH GIÁ CHỈ SỐ SINH TRẮC HỌC")
    print(f"Tổng số cặp ảnh đã đối soát: {len(pair_labels):,}")
    print(f"Độ chính xác cặp (Pairwise Accuracy): {accuracy:.2f}%")
    print(f"Ngưỡng Cosine tối ưu (Threshold): {optimal_threshold:.4f}")
    print(f"FAR (Chấp nhận sai) tại ngưỡng: {far[idx_target_far]:.6f}")
    print(f"FRR (Từ chối sai) tại FAR={target_far}: {frr_at_target_far * 100:.2f}%")
    print("-" * 50)
    return {
        "pair_count": int(len(pair_labels)),
        "accuracy": float(accuracy),
        "threshold": float(optimal_threshold),
        "far": float(far[idx_target_far]),
        "frr": float(frr_at_target_far),
        "target_far": float(target_far),
    }


def resolve_eval_root(dataset_root: Path) -> Path:
    split_dirs = resolve_dataset_split_dirs(dataset_root)
    if "test" in split_dirs:
        return split_dirs["test"]
    if "val" in split_dirs:
        return split_dirs["val"]
    if "all" in split_dirs:
        return split_dirs["all"]
    return split_dirs["train"]


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Khởi động trình đánh giá trên thiết bị: {device}")

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {args.dataset_root}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {args.checkpoint}")

    eval_root = resolve_eval_root(args.dataset_root)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    test_dataset = datasets.ImageFolder(root=str(eval_root), transform=transform)
    if len(test_dataset) < 2:
        raise ValueError("Dataset đánh giá phải có ít nhất 2 ảnh.")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    config = build_edgeface_config_from_metadata(checkpoint if isinstance(checkpoint, dict) else None)

    model = EdgeFaceXXS(
        embedding_dim=config.embedding_dim,
        width_preset=config.width_preset,
        stage_channels=config.stage_channels,
        rank_ratio=config.rank_ratio,
    )
    if args.apply_pruning:
        print("✂️ Áp dụng pruning mask tương thích trước khi nạp trọng số...")
        model = apply_structured_pruning(model, prune_ratio=float(checkpoint.get("prune_ratio", 0.01)))

    model.load_state_dict(state_dict)
    model = model.to(device)
    print("✅ Đã nạp trọng số thành công.")

    hardware_metrics = measure_complexity_and_latency(model, device)
    biometric_metrics = measure_biometrics(model, test_dataloader, device, target_far=args.target_far)
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "eval_root": str(eval_root),
        "model_config": config.to_metadata(),
        "hardware": hardware_metrics,
        "biometrics": biometric_metrics,
    }

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"📝 Đã lưu báo cáo tại: {args.report_json}")


if __name__ == "__main__":
    main()
