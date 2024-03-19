import torch
from torch import nn
from typing import Optional


class AttentionBlock(nn.Module):
    """
    和Transformer中的多头注意力机制的原理以及实现方式一致
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """

        :param n_channels: 等待做注意力操作特征图的通道数
        :param n_heads: 注意力头的数量
        :param d_k: 每一个注意力头处理向量的维度
        :param n_groups: Group Norm的超参数
        """
        super().__init__()

        # 一般来言d_k=num_channels//num_heads,要保证num_channels可以被num_heads整除
        if d_k is None:
            d_k = n_channels

        # 定义Group Norm层
        self.norm = nn.GroupNorm(n_groups, n_channels)

        # 多头注意力层，定义输入token和q,k,v矩阵相乘后的结果
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)

        # MLP层
        self.output = nn.Linear(n_heads * d_k, n_channels)

        self.scale = d_k ** -0.5  # 求d_k平方根的倒数
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """

        :param x: 输入数据xt，尺寸大小为(batch_size, in_channels, height, width)
        :param t: 输入数据t，尺寸大小为(batch_size, time_c)
        :return:
        """

        # 其实并没有用到t
        _ = t

        # 获取shape
        batch_size, n_channels, height, width = x.shape

        # 修改输入数据的形状，将形状修改为(batch_size,height*width,num_channels)
        # 这三个维度分别等同于Transformer中的(batch_size,seq_length,token_embedding)
        # 其实就是[batch_size,number_of_queries,query_size]
        x = x.reshape(batch_size, height * width, -1)
        # 做完self.projection(x),形状变为(batch_size,height*width,n_heads*d_k*3)
        qkv = self.projection(x).reshape(batch_size, -1, self.n_heads, 3 * self.d_k)
        # 把结果给切开 q,k,v的形状都是[batch_size,height*weight,num_heads,d_k]
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 计算注意力机制的过程
        # 想一下batch_size是不动的，对于q来讲有i个query，h个头，每个头维度是d
        # 对v来讲，有j个value，h个头，每个头的维度是d
        # 那么计算q和v的相似度其实是每个query和每个value的每个头计算一下相似度
        # 结果是[batch_size,number_of_queries,number_of_values,number_of_heads]
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale

        # 所以这里是沿着第二维度做softmax操作
        attn = attn.softmax(dim=2)

        # 接下来得到相似度之后就要根据softmax的结果加权了
        # 对于每个value按照number_of_head中的权重加权
        # 也就是batch_size不动，number_of_queries不动,value按照相似度加权
        res = torch.einsum('bijh,bjhd->bihd', attn, v)

        res = res.reshape(batch_size, -1, self.n_heads * self.d_k)

        # 再过一层MLP
        res = self.output(res)

        # 做个残差连接
        res += x

        res = res.permute(0, 2, 1).reshape(batch_size, n_channels, height, width)

        return res