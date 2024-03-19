import torch
from torch import nn
import math
from utils.activate import Swish


class TimeEmbedding(nn.Module):
    """
    该模块把整型数t，按照Transformer函数式的编码方式映射成向量，向量的形状为(batch,time_channel)，这个time_channel其实就是想把t映射成向量的维度
    """

    def __init__(self, n_channels: int):
        """

        :param n_channels: n_channels就是time_channels
        """
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        """
        Params:
            t: 维度（batch_size），整型时刻t
        """
        # 以下转换方法和Transformer的位置编码一致
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        # 输出维度(batch_size, time_channels)
        return emb