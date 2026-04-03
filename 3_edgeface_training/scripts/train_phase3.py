import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch import amp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import các module từ cấu trúc thư mục của bạn
# (Giả định các class kiến trúc đã được định nghĩa trong models/)
from models.edgeface_xxs import EdgeFaceXXS
from models.resnet101_teacher import ResNet101Teacher
from core_losses.adaface_loss import AdaFaceLoss
from core_losses.kd_loss import EmbeddingKDLoss


def compute_classification_logits(embeddings: torch.Tensor, classifier_weights: torch.Tensor) -> torch.Tensor:
    normalized_embeddings = F.normalize(embeddings.float(), p=2, dim=1)
    normalized_weights = F.normalize(classifier_weights.float(), p=2, dim=1)
    return F.linear(normalized_embeddings, normalized_weights)


def main():
    # 1. Cấu hình thiết bị (Hỗ trợ Mac M1 MPS, Cloud CUDA, hoặc CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps") # Tối ưu cho chip Apple Silicon 
    else:
        device = torch.device("cpu")
    print(f"🚀 Đang huấn luyện trên thiết bị: {device}")

    # 2. Cấu hình Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    DATA_DIR = WORKSPACE_ROOT / "2_face_dataset"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 3. Nạp dữ liệu (Data Loading & Augmentation)
    # Ảnh đã được align 112x112 từ giai đoạn 2
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Kháng nhiễu ánh sáng tại cửa
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset_full = datasets.ImageFolder(root=str(DATA_DIR), transform=train_transform)
    val_dataset_full = datasets.ImageFolder(root=str(DATA_DIR), transform=val_transform)
    num_classes = len(train_dataset_full.classes)
    total_samples = len(train_dataset_full)
    val_size = max(1, int(total_samples * 0.2))
    train_size = total_samples - val_size
    if train_size <= 0:
        raise ValueError("Dataset quá nhỏ để tách validation.")

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_samples, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(
        f"📁 Tổng số học sinh (Classes): {num_classes} | Tổng ảnh: {total_samples} | "
        f"Train: {len(train_dataset)} | Val: {len(val_dataset)}"
    )

    # Hàm hỗ trợ "tẩy rửa" keys để chống lỗi 0 keys khớp
    def load_matched_weights(model, weight_path):
        if not os.path.exists(weight_path):
            print(f"⚠️ CẢNH BÁO: Không tìm thấy file {weight_path}")
            return model

        print(f"📥 Đang xử lý file weights: {weight_path}")
        checkpoint = torch.load(weight_path, map_location='cpu')

        # Bóc tách state_dict nếu file là một checkpoint tổng hợp
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Xóa tiền tố 'module.' do DataParallel sinh ra
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v

        # Nạp vào mô hình với strict=False
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Đã nạp thành công weights vào kiến trúc!")
        return model

    # 4. Khởi tạo Mô hình (Student & Teacher)
    student_model = EdgeFaceXXS(embedding_size=512)
    teacher_model = ResNet101Teacher(embedding_size=512)

    WEIGHTS_DIR = PROJECT_ROOT / "weights"
    STUDENT_WEIGHTS = os.path.join(WEIGHTS_DIR, "edgeface_xxs.pt") # Lưu ý đuôi file của bạn đang là .pt
    TEACHER_WEIGHTS = os.path.join(WEIGHTS_DIR, "resnet101_adaface.pt")

    # Gọi hàm làm sạch và nạp weights
    student_model = load_matched_weights(student_model, STUDENT_WEIGHTS)
    teacher_model = load_matched_weights(teacher_model, TEACHER_WEIGHTS)

    # Chuyển lên M1
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 5. Khởi tạo Loss và Optimizer
    adaface_criterion = AdaFaceLoss(embedding_size=512, num_classes=num_classes).to(device)
    kd_criterion = EmbeddingKDLoss(alpha=500.0).to(device)

    optimizer = optim.AdamW([
        {'params': student_model.parameters(), 'weight_decay': 5e-4},
        {'params': adaface_criterion.parameters(), 'weight_decay': 5e-4}
    ], lr=LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Sửa lỗi FutureWarning: Xác định device string cho AMP
    device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    # GradScaler thường chỉ cần thiết cho CUDA, với MPS ta có thể tắt để tránh lỗi
    scaler = amp.GradScaler(device_type, enabled=(device_type == 'cuda'))

    # 6. VÒNG LẶP HUẤN LUYỆN
    for epoch in range(EPOCHS):
        student_model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        train_cosine_sum = 0.0
        train_cosine_batches = 0

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Sửa lỗi FutureWarning của autocast
            is_cuda = (device_type == 'cuda')
            with amp.autocast(device_type=device_type, enabled=is_cuda):
                # Trích xuất đặc trưng từ Học sinh
                student_embeddings, student_norms = student_model(images)
                
                # Trích xuất đặc trưng từ Giáo viên (Không tính gradient)
                with torch.no_grad():
                    teacher_embeddings, _ = teacher_model(images)

                # Tính toán Loss tổng hợp [cite: 31]
                loss_adaface = adaface_criterion(student_embeddings, student_norms, labels)
                loss_kd = kd_criterion(student_embeddings, teacher_embeddings)
                loss = loss_adaface + loss_kd

            # Lan truyền ngược với GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            logits = compute_classification_logits(student_embeddings.detach(), adaface_criterion.weights.detach())
            predictions = logits.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            batch_cosine = F.cosine_similarity(
                student_embeddings.detach().float(),
                teacher_embeddings.detach().float(),
                dim=1
            ).mean().item()
            train_cosine_sum += batch_cosine
            train_cosine_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(train_dataloader)}] | "
                      f"Loss: {loss.item():.4f} (AdaFace: {loss_adaface.item():.4f}, KD: {loss_kd.item():.4f})")

        train_accuracy = 100.0 * train_correct / max(1, train_total)
        train_cosine = train_cosine_sum / max(1, train_cosine_batches)

        student_model.eval()
        val_correct = 0
        val_total = 0
        val_cosine_sum = 0.0
        val_cosine_batches = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                student_embeddings, _ = student_model(images)
                teacher_embeddings, _ = teacher_model(images)
                logits = compute_classification_logits(student_embeddings, adaface_criterion.weights)
                predictions = logits.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                batch_cosine = F.cosine_similarity(
                    student_embeddings.float(),
                    teacher_embeddings.float(),
                    dim=1
                ).mean().item()
                val_cosine_sum += batch_cosine
                val_cosine_batches += 1

        val_accuracy = 100.0 * val_correct / max(1, val_total)
        val_cosine = val_cosine_sum / max(1, val_cosine_batches)

        scheduler.step()
        print(
            f"🎉 Hết Epoch {epoch+1} | Trung bình Loss: {total_loss/len(train_dataloader):.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}% | "
            f"Train Cosine: {train_cosine:.4f} | Val Cosine: {val_cosine:.4f}"
        )

        # Lưu checkpoint sau mỗi epoch
        torch.save(student_model.state_dict(), os.path.join(CHECKPOINT_DIR, f"phase3_base_model_ep{epoch+1}.pth"))

    print("🎉 Hoàn tất Giai đoạn 3! Trọng số đã được lưu.")
    # Lưu trọng số cuối cùng chuẩn bị cho Giai đoạn 4
    torch.save(student_model.state_dict(), CHECKPOINT_DIR / "phase3_base_model.pth")

if __name__ == "__main__":
    main()
