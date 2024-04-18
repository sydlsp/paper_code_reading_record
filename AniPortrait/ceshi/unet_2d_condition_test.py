import torch

from src.models.unet_2d_condition import UNet2DConditionModel
path=path='/share/ckpt/shipu/ani/pretrained_model/reference_unet.pth'
unet=UNet2DConditionModel.from_pretrained(
        path,
        subfolder="unet",
    ).to(dtype=torch.float16, device="cuda")