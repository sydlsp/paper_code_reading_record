import torch
from torch import nn


class DownSample(nn.Module):
    """
    下采样：其实是特征压缩，卷积
    """

    def __init__(self, n_channels):
        super().__init__()
        # out_size=(in_size-3+2*2)/2+1，这里是把图像尺寸减小一般
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)


class Upsample(nn.Module):
    """
    上采样，主要作用是还原特征，反卷积
    """

    def __init__(self, n_channels):
        super().__init__()
        # out_size=(in_size-1)*stride+kernel_size,-2*padding+output_padding
        # out_size=(in_size-1)*2+4-2*1+0，其实就是把图像尺寸扩大2倍
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)