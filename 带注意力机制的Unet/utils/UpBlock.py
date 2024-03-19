import torch
from torch import nn
from utils.Attention import AttentionBlock
from utils.ResidualBlock import ResidualBlock


# %%
class UpBlock(nn.Module):
    """
    这一模块也是有残差块和注意力块结合而成的
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()

        # 看Unet的示意图右边的部分是要和左边的结果接起来的
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):

        x = self.res(x, t)
        x = self.attn(x)
        return x