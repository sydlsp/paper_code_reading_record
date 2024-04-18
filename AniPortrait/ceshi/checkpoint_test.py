import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as cp

net=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,256))

x=torch.randn(size=[32,64],requires_grad=True)

x_out=cp(net,x)
print(x_out.shape)