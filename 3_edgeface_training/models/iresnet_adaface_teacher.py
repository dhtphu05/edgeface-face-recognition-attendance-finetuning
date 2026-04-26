from __future__ import annotations

import torch
from torch import nn


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.shortcut_layer = None
        if stride != 1 or inplanes != planes:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x if self.shortcut_layer is None else self.shortcut_layer(x)
        return self.res_layer(x) + shortcut


class IResNetAdaFaceBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 512, dropout: float = 0.4) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )

        blocks = [
            (64, 3, 1),
            (128, 13, 2),
            (256, 30, 2),
            (512, 3, 2),
        ]
        body_layers: list[nn.Module] = []
        inplanes = 64
        for planes, num_units, stride in blocks:
            body_layers.append(IBasicBlock(inplanes, planes, stride=stride))
            inplanes = planes
            for _ in range(num_units - 1):
                body_layers.append(IBasicBlock(inplanes, planes, stride=1))
        self.body = nn.Sequential(*body_layers)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(p=dropout),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, embedding_dim),
            nn.BatchNorm1d(embedding_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.body(x)
        return self.output_layer(x)


class IResNet101AdaFaceTeacher(nn.Module):
    def __init__(self, embedding_dim: int | None = None, embedding_size: int | None = None) -> None:
        super().__init__()
        if embedding_dim is None:
            embedding_dim = embedding_size if embedding_size is not None else 512
        elif embedding_size is not None and embedding_size != embedding_dim:
            raise ValueError("embedding_dim and embedding_size must match when both are provided.")
        self.net = IResNetAdaFaceBackbone(embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.net(x)
        norms = embeddings.norm(p=2, dim=1)
        return embeddings, norms
