import torch
from torch import nn
from utils.ResidualBlock import ResidualBlock
from utils.Attention import AttentionBlock


class MiddleBlock(nn.Module):
    """
    中间块是 残差块+注意力+残差块构建而成的
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x

