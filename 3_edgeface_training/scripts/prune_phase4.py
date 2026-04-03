import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import kiến trúc mạng đã gắn LoRaLin của bạn
from models.edgeface_xxs import EdgeFaceXXS

def calculate_l1_norm(weight_tensor):
    """
    Tính toán L1-norm cho từng bộ lọc (filter) trong một lớp Conv2d.
    weight_tensor có shape: (out_channels, in_channels, kernel_size, kernel_size)
    """
    # Tính tổng giá trị tuyệt đối dọc theo các chiều (trừ chiều out_channels)
    return torch.sum(torch.abs(weight_tensor), dim=(1, 2, 3))

def apply_structured_pruning(model, prune_ratio=0.05):
    """
    Thực hiện cắt tỉa cấu trúc 5% các kênh (channels) yếu nhất.
    Chỉ tập trung vào các lớp ở phần sâu của mạng (như Stage 3 & 4) 
    để bảo toàn các đặc trưng cơ sở ở các lớp đầu.
    """
    pruned_layers_count = 0
    
    for name, module in model.named_modules():
        # Lọc ra các lớp Convolution 2D thông thường (không phải pointwise 1x1 đã thay bằng LoRaLin)
        if isinstance(module, nn.Conv2d) and module.groups == 1:
            # Bỏ qua lớp Conv đầu tiên (thường chứa đặc trưng cạnh, màu sắc rất quan trọng)
            if 'stage1' in name or 'stem' in name:
                continue
                
            # Sử dụng module prune có sẵn của PyTorch để loại bỏ theo L1-norm
            # Tham số n=1 tương đương với L1-norm, dim=0 nghĩa là cắt bỏ dọc theo out_channels
            prune.ln_structured(
                module, 
                name='weight', 
                amount=prune_ratio, 
                n=1, 
                dim=0
            )
            
            # Xóa bỏ các trọng số đã bị mask (biến việc cắt tỉa ảo thành thật để giảm dung lượng)
            prune.remove(module, 'weight')
            pruned_layers_count += 1
            
            print(f"✂️ Đã cắt tỉa {prune_ratio*100}% kênh tại lớp: {name}")

    print(f"Tổng cộng đã cắt tỉa {pruned_layers_count} lớp Tích chập.")
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Bắt đầu quá trình Cắt tỉa (Pruning) trên thiết bị: {device}")

    # 1. Đường dẫn thư mục
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    BASE_MODEL_PATH = CHECKPOINT_DIR / "phase3_base_model.pth"
    PRUNED_MODEL_PATH = CHECKPOINT_DIR / "phase4_pruned_model.pth"

    # 2. Khởi tạo mô hình và nạp trọng số từ Giai đoạn 3
    model = EdgeFaceXXS(embedding_size=512).to(device)
    
    try:
        model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
        print("✅ Đã nạp thành công trọng số gốc từ Giai đoạn 3.")
    except Exception as e:
        print(f"❌ Lỗi khi nạp mô hình: {e}")
        return

    # 3. Đánh giá số lượng tham số TRƯỚC khi cắt tỉa
    total_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Số lượng tham số ban đầu: {total_params_before:,}")

    # 4. Áp dụng Cắt tỉa Cấu trúc (Tỉ lệ 5%)
    # Lưu ý: Việc cắt 5% mỗi chu kỳ (Iterative Pruning) an toàn hơn cắt 25% một lần
    model = apply_structured_pruning(model, prune_ratio=0.05)

    # 5. Lưu lại mô hình đã bị cắt tỉa
    # Mô hình lúc này đã bị "tổn thương" nhẹ, cần phải mang sang Phase 5 để Fine-tune phục hồi
    torch.save(model.state_dict(), PRUNED_MODEL_PATH)
    print(f"💾 Đã lưu mô hình sau khi cắt tỉa tại: {PRUNED_MODEL_PATH}")
    print("⚠️ CẢNH BÁO: Độ chính xác hiện tại đang giảm nhẹ. Hãy chạy kịch bản Fine-tune (Phase 5) ngay!")

if __name__ == "__main__":
    main()
