import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaFaceLoss(nn.Module):
    def __init__(self, embedding_size=512, num_classes=1000, margin=0.4, scale=64.0):
        super(AdaFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        # Ma trận trọng số đại diện cho các Identity (Học sinh)
        self.weights = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weights)
        
        # Sử dụng BatchNorm1d(affine=False) như một cơ chế tính EMA tự động cho Norm
        self.batch_norm = nn.BatchNorm1d(1, affine=False, momentum=0.01)

    def forward(self, embeddings, norms, labels):
        # 1. L2 Normalize weights và embeddings
        normalized_weights = F.normalize(self.weights, p=2, dim=1)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 2. Tính Cosine Similarity (Góc giữa vector khuôn mặt và vector nhãn)
        cosine = F.linear(normalized_embeddings, normalized_weights)
        
        # 3. Tính toán chất lượng ảnh (Quality Indicator) thông qua Norm có EMA
        norms_ema = self.batch_norm(norms.unsqueeze(1)).squeeze(1) # [Batch_size]
        # Giới hạn chất lượng trong khoảng [-1, 1]
        quality_indicator = torch.clamp(norms_ema, -1.0, 1.0)
        
        # 4. Điều chỉnh lề (Adaptive Margin)
        # Ảnh mờ (quality < 0): giảm margin. Ảnh nét (quality > 0): tăng margin.
        g_angular = self.margin * quality_indicator
        g_additive = self.margin + (self.margin * quality_indicator * -1)
        
        # 5. Áp dụng lề vào Cosine Similarity
        cosine_m = torch.cos(torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)) + g_angular.unsqueeze(1)) - g_additive.unsqueeze(1)
        
        # 6. Tạo ma trận One-hot và thay thế các giá trị cosine mục tiêu
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        
        output = (one_hot * cosine_m) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        # Trả về CrossEntropy tiêu chuẩn sau khi đã can thiệp lề
        return F.cross_entropy(output, labels)