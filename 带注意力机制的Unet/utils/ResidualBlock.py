import torch
from torch import nn
from utils.activate import Swish


class ResidualBlock(nn.Module):
    """
    每个ResidualBlock都有两层CNN做特征提取
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 1,
                 dropout: float = 0.1):
        """
        :param in_channels: 输入图片的Channel数量
        :param out_channels: 经过残差块后输出的channel数量
        :param time_channels:时间步的编码长度
        :param n_groups: GroupNorm的超参数
        :param dropout: dropout概率
        """
        super().__init__()

        # 第一层卷积
        self.norm1 = nn.GroupNorm(n_groups, in_channels)  # 组归一化方式，将in_channels分成n_groups组然后再归一化
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))  # 这样的卷积设置不改变图像大小，只改变通道数

        # 第二层卷积是类似的
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # 当in_channels与out_channels，残差连接直接将输入输出相加
        # 否则的话对输入做一次卷积
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()  # 对输入不做任何操作直接输出

        # 时间步t的编码维度有可能不等于out_channels所以要对时间步编码先进行一次线性转换
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """

        :param x: 输入数据xt，尺寸大小为(batch_size, in_channels, height, width)
        :param t: 输入数据t，尺寸大小为(batch_size, time_c)
        """

        # 输入图片先过第一层卷积
        h = self.conv1(self.act1(self.norm1(x)))

        # 先对图片做激活和线性变换，将时间步编码变为out_channels长度，然后将对应的数字和图片对应的channel相加
        time_embedding = self.time_emb(self.time_act(t))
        time_embedding = time_embedding.reshape(time_embedding.shape[0], time_embedding.shape[1], 1, 1)
        #print(time_embedding.shape)
        h += time_embedding

        # 对图片再做一次卷积

        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # 处理一下原始输入
        x_res = self.shortcut(x)

        # 返回最终结果
        return h + x_res