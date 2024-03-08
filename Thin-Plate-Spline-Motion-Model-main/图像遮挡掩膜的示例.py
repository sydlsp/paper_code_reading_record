import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

img=plt.imread("th.jpg")
print(img.shape)
plt.imshow(img)
plt.show()
aug=torchvision.transforms.ToTensor()
img=aug(img)
print(img.shape)

a,b,c=img.shape
mask=torch.zeros(size=(b,c))
mask[:200,:200]=1
print(mask)

img=img*mask
print("ok"+str(img.shape))
img=img.permute(1,2,0)
print(img.shape)
plt.imshow(img)
plt.show()

