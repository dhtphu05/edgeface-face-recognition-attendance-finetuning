import torch
import torch.nn as nn


class LoRaLinConv1x1(nn.Module):
    """
    Module phân rã hạng thấp (Low-Rank) tối ưu riêng cho lớp Pointwise Conv (1x1).
    Bảo toàn cấu trúc không gian H x W của ma trận ảnh.
    """
    def __init__(self, in_channels, out_channels, stride=1, rank_ratio=0.6):
        super(LoRaLinConv1x1, self).__init__()
        if rank_ratio <= 0:
            raise ValueError("rank_ratio must be greater than 0.")

        # Tính toán hạng (rank) trung gian r
        self.rank = max(2, int(min(in_channels, out_channels) * rank_ratio))
        self.rank_ratio = rank_ratio

        # Phân rã Conv 1x1 thành lớp Nén (Compress) và Giải nén (Expand)
        self.compress = nn.Conv2d(in_channels, self.rank, kernel_size=1, stride=stride, bias=False)
        self.expand = nn.Conv2d(self.rank, out_channels, kernel_size=1, stride=1, bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.compress(x)
        x = self.expand(x)
        x = self.bn(x)
        return self.act(x)

def replace_conv1x1_with_loralin(model, rank_ratio=0.6):
    """
    Hàm đệ quy quét qua toàn bộ kiến trúc mạng và tự động ghi đè
    các lớp Conv2d(kernel_size=1) bằng LoRaLinConv1x1.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
            loralin_layer = LoRaLinConv1x1(
                in_channels=module.in_channels, 
                out_channels=module.out_channels, 
                stride=module.stride[0], 
                rank_ratio=rank_ratio
            )
            setattr(model, name, loralin_layer)
        else:
            replace_conv1x1_with_loralin(module, rank_ratio)
    return model


# Backward-compatible alias used by older scripts/modules.
LoRaLinConv = LoRaLinConv1x1
