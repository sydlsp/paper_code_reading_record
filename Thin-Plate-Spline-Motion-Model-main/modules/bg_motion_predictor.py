from torch import nn
import torch
from torchvision import models

"""
作者假设从原图像背景到驱动图像背景的变换满足仿射变换
仿射变换的矩阵形式是最后一行满足[0 0 1]
那么需要有剩下6个参数信息，作者采用ResNet网络来拟合这六个参数
关于仿射变换的更多信息可参考：https://zhuanlan.zhihu.com/p/669100382#:~:text=%E4%BB%BF%E5%B0%84%E5%8F%98%E6%8D%A2%E7%9F%A9%E9%98%B5%E8%A1%A8%E7%A4%BA%E5%90%84%E9%83%A8%E5%88%86%E6%8F%8F%E8%BF%B0,%E7%9F%A9%E9%98%B5T%20%282%C3%973%29%E5%B0%B1%E7%A7%B0%E4%B8%BA%E4%BB%BF%E5%B0%84%E5%8F%98%E6%8D%A2%E7%9A%84%E5%8F%98%E6%8D%A2%E7%9F%A9%E9%98%B5%EF%BC%8CR%E4%B8%BA%E7%BA%BF%E6%80%A7%E5%8F%98%E6%8D%A2%E7%9F%A9%E9%98%B5%EF%BC%8Ct%E4%B8%BA%E5%B9%B3%E7%A7%BB%E7%9F%A9%E9%98%B5%EF%BC%8C%E7%AE%80%E5%8D%95%E6%9D%A5%E8%AF%B4%EF%BC%8C%E4%BB%BF%E5%B0%84%E5%8F%98%E6%8D%A2%E5%B0%B1%E6%98%AF%E7%BA%BF%E6%80%A7%E5%8F%98%E6%8D%A2%2B%E5%B9%B3%E7%A7%BB%E3%80%82
"""
class BGMotionPredictor(nn.Module):
    """
    Module for background estimation, return single transformation, parametrized as 3x3 matrix. The third row is [0 0 1]
    """

    def __init__(self):
        super(BGMotionPredictor, self).__init__()
        self.bg_encoder = models.resnet18(pretrained=False)
        #这里是把最开始卷积层的通道数增加了
        self.bg_encoder.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_features = self.bg_encoder.fc.in_features
        self.bg_encoder.fc = nn.Linear(num_features, 6)
        #将最后一层的权重置为0
        self.bg_encoder.fc.weight.data.zero_()
        self.bg_encoder.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, source_image, driving_image):
        bs = source_image.shape[0]
        #先生成一个3行3列的只有对角线元素为1其他都为0的矩阵，将其升维到[1,3,3]再沿着第0维复制batch_size遍
        out = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).type(source_image.type())
        #将原图像和驱动图像连接放到ResNet中，每张图片最后编码成一个6维向量
        prediction = self.bg_encoder(torch.cat([source_image, driving_image], dim=1))
        #这个过程是说对于每张图片来言将6维向量转化为2*3
        #然后将值赋值给out矩阵，这样达到对于每个对角矩阵来言，其上面两行被赋值了
        #最后一行0 0 1保持不变
        #这就是放射变换的矩阵形式
        out[:, :2, :] = prediction.view(bs, 2, 3)
        return out
