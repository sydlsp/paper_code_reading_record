from torch import nn
import torch
from torchvision import models

class KPDetector(nn.Module):
    """
    Predict K*5 keypoints.
    """
    #有num_teps个TPS变换
    def __init__(self, num_tps, **kwargs):
        super(KPDetector, self).__init__()
        self.num_tps = num_tps

        #修改resnet18最后一层分类层的输出
        self.fg_encoder = models.resnet18(pretrained=False)
        num_features = self.fg_encoder.fc.in_features
        #这里有个2的原因是关键点(x,y)需要两个数字表示
        self.fg_encoder.fc = nn.Linear(num_features, num_tps*5*2)

        
    def forward(self, image):

        #对图片用resnet网络编码
        fg_kp = self.fg_encoder(image)
        bs, _, = fg_kp.shape
        fg_kp = torch.sigmoid(fg_kp)
        #感觉这里是做完激活后又加了个线性层，只是这个线性层的参数是固定的
        fg_kp = fg_kp * 2 - 1
        out = {'fg_kp': fg_kp.view(bs, self.num_tps*5, -1)}

        return out
