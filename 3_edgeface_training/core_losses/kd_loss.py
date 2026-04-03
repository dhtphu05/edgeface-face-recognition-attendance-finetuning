import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingKDLoss(nn.Module):
    """
    Chưng cất tri thức cấp độ thực thể (Instance-Level Embedding Distillation).
    Ép vector đặc trưng của EdgeFace (Student) bám sát ResNet-101 (Teacher).
    """
    def __init__(self, alpha=10.0):
        super(EmbeddingKDLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha # Trọng số cân bằng cho hàm KD

    def forward(self, student_embeddings, teacher_embeddings):
        # Normalize để đưa cả 2 vector về cùng một không gian hình cầu (Hypersphere)
        student_norm = F.normalize(student_embeddings.float(), p=2, dim=1)
        
        # Không cần tính đạo hàm cho Teacher để tiết kiệm VRAM
        with torch.no_grad():
            teacher_norm = F.normalize(teacher_embeddings.float(), p=2, dim=1)
            
        # Tính Mean Squared Error giữa 2 không gian nhúng
        loss = self.mse_loss(student_norm, teacher_norm)
        
        return loss * self.alpha


def distillation_loss(student_embeddings, teacher_embeddings, alpha=10.0):
    """Backward-compatible functional wrapper used by older imports."""
    criterion = EmbeddingKDLoss(alpha=alpha)
    return criterion(student_embeddings, teacher_embeddings)
