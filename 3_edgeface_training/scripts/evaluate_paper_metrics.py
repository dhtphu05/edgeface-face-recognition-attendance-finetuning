import os
import sys
import time
from pathlib import Path
import torch
import numpy as np
from thop import profile, clever_format
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve
from scipy.spatial.distance import pdist, squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import kiến trúc mạng của bạn (Đảm bảo kiến trúc đã được định nghĩa đúng)
from models.edgeface_xxs import EdgeFaceXXS
# Nếu đánh giá mô hình Giai đoạn 5, cần import hàm apply_structured_pruning
from scripts.prune_phase4 import apply_structured_pruning

def measure_complexity_and_latency(model, device):
    """Đo lường FLOPs, Tham số và Tốc độ suy luận (Latency)"""
    model.eval()
    
    # 1. Đo lường FLOPs và Params với dummy input 112x112
    dummy_input = torch.randn(1, 3, 112, 112).to(device)
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    
    # MFLOPs (Mega FLOPs) = MACs * 2 / 10^6 (Ước tính tương đối)
    flops = macs * 2
    flops_str, params_str = clever_format([flops, params], "%.2f")
    
    print("-" * 50)
    print("📊 ĐÁNH GIÁ ĐỘ PHỨC TẠP VÀ TỐC ĐỘ PHẦN CỨNG")
    print(f"Tổng số Tham số (Params): {params_str}")
    print(f"Tổng khối lượng tính toán (FLOPs): {flops_str}")
    
    # 2. Đo lường Latency (Warm-up 100 lần, Đo 1000 lần)
    print("⏳ Đang đo lường Latency (Giả lập suy luận 1000 lần)...")
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input) # Warm-up GPU/CPU
            
        start_time = time.time()
        for _ in range(1000):
            _ = model(dummy_input)
        end_time = time.time()
        
    avg_latency = ((end_time - start_time) / 1000) * 1000 # Đổi ra mili-giây (ms)
    fps = 1000 / avg_latency
    print(f"Độ trễ trung bình (Latency): {avg_latency:.2f} ms / khung hình")
    print(f"Tốc độ khung hình (FPS): {fps:.2f} FPS")
    print("-" * 50)

def measure_biometrics(model, dataloader, device):
    """Đo lường FAR, FRR và Độ chính xác (Accuracy) trên tập dữ liệu kiểm thử"""
    model.eval()
    embeddings_list = []
    labels_list = []
    
    print("🔍 Đang trích xuất đặc trưng (Embeddings) cho toàn bộ tập Test...")
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            # Lấy vector nhúng (bỏ qua norms nếu model trả về 2 output)
            outputs = model(images)
            if isinstance(outputs, tuple):
                embeddings = outputs[0]
            else:
                embeddings = outputs
                
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.append(labels.numpy())
            
    # Gộp toàn bộ dữ liệu
    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list)
    
    # 1. Tính toán ma trận khoảng cách Cosine cho TẤT CẢ các cặp ảnh
    print("🧮 Đang tính toán ma trận Cosine Similarity...")
    # pdist tính khoảng cách, (1 - khoảng cách) = độ tương đồng Cosine
    cosine_dists = pdist(embeddings, metric='cosine')
    similarities = 1 - cosine_dists
    
    # 2. Tạo nhãn cặp (Pair Labels): 1 nếu cùng người (Positive), 0 nếu khác người (Negative)
    # So sánh mọi cặp label xem có giống nhau không
    n_samples = len(labels)
    pair_labels = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            pair_labels.append(1 if labels[i] == labels[j] else 0)
    pair_labels = np.array(pair_labels)
    
    # 3. Sử dụng ROC Curve để tìm FAR, FRR
    print("📈 Đang phân tích đường cong ROC...")
    fpr, tpr, thresholds = roc_curve(pair_labels, similarities)
    
    # FRR (False Rejection Rate) = 1 - TPR (True Positive Rate)
    frr = 1 - tpr
    # FAR (False Acceptance Rate) = FPR (False Positive Rate)
    far = fpr
    
    # Tìm ngưỡng (Threshold) tối ưu nơi FAR tiệm cận EER (Equal Error Rate) hoặc một mức FAR cố định
    # Thường trong Face Recog, người ta báo cáo FRR tại FAR = 0.001 hoặc 0.0001
    target_far = 0.001 # 10^-3
    
    # Tìm index mà tại đó FAR gần với target_far nhất
    idx_target_far = np.abs(far - target_far).argmin()
    
    optimal_threshold = thresholds[idx_target_far]
    frr_at_target_far = frr[idx_target_far]
    
    # Tính Accuracy đơn giản tại ngưỡng này
    predictions = (similarities >= optimal_threshold).astype(int)
    accuracy = np.mean(predictions == pair_labels) * 100
    
    print("🧬 ĐÁNH GIÁ CHỈ SỐ SINH TRẮC HỌC")
    print(f"Tổng số cặp ảnh đã đối soát: {len(pair_labels):,}")
    print(f"Độ chính xác cặp (Pairwise Accuracy): {accuracy:.2f}%")
    print(f"Ngưỡng Cosine tối ưu (Threshold): {optimal_threshold:.4f}")
    print(f"FAR (Chấp nhận sai) tại ngưỡng: {far[idx_target_far]:.6f}")
    print(f"FRR (Từ chối sai) tại FAR={target_far}: {frr_at_target_far*100:.2f}%")
    print("-" * 50)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Khởi động trình đánh giá trên thiết bị: {device}")
    
    # 1. Cấu hình
    # CHÚ Ý: Hãy trỏ đến thư mục Test Set của bạn (có thể lấy ra 20% từ tập 2_face_dataset)
    TEST_DATA_DIR = WORKSPACE_ROOT / "2_face_dataset"
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "phase5_final_model.pth"
    
    # 2. Nạp dữ liệu kiểm thử
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_dataset = datasets.ImageFolder(root=str(TEST_DATA_DIR), transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 3. Nạp mô hình
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        embedding_dim = checkpoint.get("embedding_dim", 256)
        state_dict = checkpoint["model_state_dict"]
    else:
        embedding_dim = 256
        state_dict = checkpoint

    model = EdgeFaceXXS(embedding_dim=embedding_dim)
    
    # NẾU BẠN ĐANG ĐO MÔ HÌNH Ở GIAI ĐOẠN 5 (Đã cắt tỉa):
    print("✂️ Đang thiết lập kiến trúc Pruning để khớp trọng số...")
    model = apply_structured_pruning(model, prune_ratio=0.05)
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("✅ Đã nạp trọng số thành công.")
    
    # 4. Chạy các hàm đánh giá
    measure_complexity_and_latency(model, device)
    measure_biometrics(model, test_dataloader, device)

if __name__ == "__main__":
    main()
