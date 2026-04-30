from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim: int, num_domains: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
