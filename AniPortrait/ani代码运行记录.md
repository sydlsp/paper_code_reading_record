# ani代码运行记录



从python -m scripts.pose2vid --config ./configs/prompts/animation.yaml -W 512 -H 512 -acc开始吧

首先修改了animation.yaml 相关ckpt的路径



然后在运行的过程中修改了一下frame_interpolation.py  第13行的路径信息，修改为：

```python
checkpoint_name = os.path.join("/share/ckpt/shipu/ani/pretrained_model/film_net_fp16.pt")
```





## vae运行初探

vae的输入是[batch_size,num_channels,H,W] 输出的形状和输入是相同的，从运行结果上来看，原图和编码解码后的图片有细微区别

```python
import torch
from diffusers import AutoencoderKL
import os

from PIL import Image
import matplotlib.pyplot as plt
import torchvision

#vae模型的初始化
path='/share/ckpt/shipu/ani/pretrained_model/sd-vae-ft-mse'

vae=AutoencoderKL.from_pretrained('/share/ckpt/shipu/ani/pretrained_model/sd-vae-ft-mse').to("cuda:0", dtype=torch.float16)

#图片读取
path="./test_pic.jpg"
pic= Image.open(path)
args=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
pic=args(pic)
pic=pic.unsqueeze(dim=0).half().to("cuda:0")


#pic=torch.randn(size=[1,3,512,512]).half().to("cuda:0")

#利用vae进行图片编码解码
pic_vae=vae(pic)

#图片保存
save_img=pic_vae['sample'].squeeze(dim=0)

save_img=torchvision.transforms.ToPILImage()(save_img)

save_img.save("result.jpg")
```





## Unet_2d_condition_model

### 先看一下参数含义：

```markdown
这是一个有条件的二维UNet模型，接受一个带噪声的样本、条件状态和一个时间步长，并返回一个形状为样本的输出。

这个模型继承自[ModelMixin]。查看超类文档以了解所有模型通用方法的实现（如下载或保存）。

参数:

sample_size（int或Tuple[int, int]，可选，默认为None）：输入/输出样本的高度和宽度。
in_channels（int，可选，默认为4）：输入样本中的通道数。
out_channels（int，可选，默认为4）：输出中的通道数。
center_input_sample（bool，可选，默认为False）：是否将输入样本居中。
flip_sin_to_cos（bool，可选，默认为False）：是否在时间嵌入中将sin翻转为cos。
freq_shift（int，可选，默认为0）：应用于时间嵌入的频率偏移量。
down_block_types（Tuple[str]，可选，默认为("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")）：要使用的下采样块的元组。
mid_block_type（str，可选，默认为"UNetMidBlock2DCrossAttn"）：UNet中间部分的块类型，可以是UNetMidBlock2DCrossAttn、UNetMidBlock2D或UNetMidBlock2DSimpleCrossAttn之一。如果为None，则跳过中间块层。
up_block_types（Tuple[str]，可选，默认为("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")）：要使用的上采样块的元组。
only_cross_attention（bool或Tuple[bool]，可选，默认为False）：是否在基本的transformer块中包含自注意力，参见[~models.attention.BasicTransformerBlock]。
block_out_channels（Tuple[int]，可选，默认为(320, 640, 1280, 1280)）：每个块的输出通道的元组。
layers_per_block（int，可选，默认为2）：每个块的层数。
downsample_padding（int，可选，默认为1）：用于下采样卷积的填充。
mid_block_scale_factor（float，可选，默认为1.0）：用于中间块的缩放因子。
dropout（float，可选，默认为0.0）：要使用的dropout概率。
act_fn（str，可选，默认为"silu"）：要使用的激活函数。
norm_num_groups（int，可选，默认为32）：用于规范化的组数。如果为None，则跳过后处理中的规范化和激活层。
norm_eps（float，可选，默认为1e-5）：用于规范化的epsilon。
cross_attention_dim（int或Tuple[int]，可选，默认为1280）：跨注意力特征的维度。
transformer_layers_per_block（int、Tuple[int]或Tuple[Tuple]，可选，默认为1）：[~models.attention.BasicTransformerBlock]类型的transformer块的数量。仅对[~models.unet_2d_blocks.CrossAttnDownBlock2D]、[~models.unet_2d_blocks.CrossAttnUpBlock2D]、[~models.unet_2d_blocks.UNetMidBlock2DCrossAttn]相关。
reverse_transformer_layers_per_block（Tuple[Tuple]，可选，默认为None）：在U-Net的上采样块中使用[~models.attention.BasicTransformerBlock]类型的transformer块的数量。仅当transformer_layers_per_block类型为Tuple[Tuple]时，以及对[~models.unet_2d_blocks.CrossAttnDownBlock2D]、[~models.unet_2d_blocks.CrossAttnUpBlock2D]、[~models.unet_2d_blocks.UNetMidBlock2DCrossAttn]相关。
encoder_hid_dim（int，可选，默认为None）：如果定义了encoder_hid_dim_type，则encoder_hidden_states将从encoder_hid_dim维度投影到cross_attention_dim。
encoder_hid_dim_type（str，可选，默认为None）：如果给定，则将encoder_hidden_states和可能的其他嵌入down-projected到维度cross_attention，以便根据encoder_hid_dim_type进行文本嵌入。
attention_head_dim（int，可选，默认为8）：注意力头的维度。
num_attention_heads（int，可选）：注意力头的数量。如果未定义，默认为attention_head_dim。
resnet_time_scale_shift（str，可选，默认为"default"）：ResNet块的时间尺度转移配置（参见[~models.resnet.ResnetBlock2D]）。从default或scale_shift中选择。
class_embed_type（str，可选，默认为None）：要使用的类嵌入的类型，最终将其与时间嵌入求和。从None、"timestep"、"identity"、"projection"或"simple_projection"中选择。
addition_embed_type（str，可选，默认为None）：配置一个可选的嵌入，将与时间嵌入求和。从None或"text"中选择。"text"将使用TextTimeEmbedding层。
addition_time_embed_dim（int，可选，默认为None）：时间步骤嵌入的维度。
num_class_embeds（int，可选，默认为None）：可学习的嵌入矩阵的输入维度，将被投影到time_embed_dim，当使用class_embed_type="projection"且class_embed_type="projection"时需要。
time_embedding_type（str，可选，默认为positional）：用于时间步骤的位置嵌入类型。从positional或fourier中选择。
time_embedding_dim（int，可选，默认为None）：投影时间嵌入的维度的可选覆盖。
time_embedding_act_fn（str，可选，默认为None）：仅对时间嵌入使用一次的可选激活函数，然后将其传递给UNet的其余部分。从silu、mish、gelu和swish中选择。
timestep_post_act（str，可选，默认为None）：在时间步骤嵌入中使用的第二个激活函数。从silu、mish和gelu中选择。
time_cond_proj_dim（int，可选，默认为None）：时间步骤嵌入中cond_proj层的维度。
conv_in_kernel（int，可选，默认为3）：conv_in层的核大小。
conv_out_kernel（int，可选，默认为3）：conv_out层的核大小。
projection_class_embeddings_input_dim（int，可选）：当class_embed_type="projection"时，class_labels输入的维度。在class_embed_type="projection"时需要。
class_embeddings_concat（bool，可选，默认为False）：是否将时间嵌入与类嵌入连接起来。
mid_block_only_cross_attention（bool，可选，默认为None）：在使用UNetMidBlock2DSimpleCrossAttn时，是否在中间块中使用交叉注意力。如果only_cross_attention被给定为单个布尔值，并且mid_block_only_cross_attention为None，则将only_cross_attention的值用作mid_block_only_cross_attention的值。否则默认为False。
```



### 模型结构（默认）：

```python
UNet2DConditionModel(
  (conv_in): Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (time_proj): Timesteps()
  (time_embedding): TimestepEmbedding(
    (linear_1): LoRACompatibleLinear(in_features=320, out_features=1280, bias=True)
    (act): SiLU()
    (linear_2): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=True)
  )
  (down_blocks): ModuleList(
    (0): CrossAttnDownBlock2D(
      (attentions): ModuleList(
        (0-1): 2 x Transformer2DModel(
          (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
          (proj_in): LoRACompatibleConv(320, 320, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): LoRACompatibleLinear(in_features=320, out_features=320, bias=False)
                (to_k): LoRACompatibleLinear(in_features=320, out_features=320, bias=False)
                (to_v): LoRACompatibleLinear(in_features=320, out_features=320, bias=False)
                (to_out): ModuleList(
                  (0): LoRACompatibleLinear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): LoRACompatibleLinear(in_features=320, out_features=320, bias=False)
                (to_k): LoRACompatibleLinear(in_features=768, out_features=320, bias=False)
                (to_v): LoRACompatibleLinear(in_features=768, out_features=320, bias=False)
                (to_out): ModuleList(
                  (0): LoRACompatibleLinear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): LoRACompatibleLinear(in_features=320, out_features=2560, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): LoRACompatibleLinear(in_features=1280, out_features=320, bias=True)
                )
              )
            )
          )
          (proj_out): LoRACompatibleConv(320, 320, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
          (conv1): LoRACompatibleConv(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): LoRACompatibleLinear(in_features=1280, out_features=320, bias=True)
          (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): LoRACompatibleConv(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): LoRACompatibleConv(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (1): CrossAttnDownBlock2D(
      (attentions): ModuleList(
        (0-1): 2 x Transformer2DModel(
          (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
          (proj_in): LoRACompatibleConv(640, 640, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): LoRACompatibleLinear(in_features=640, out_features=640, bias=False)
                (to_k): LoRACompatibleLinear(in_features=640, out_features=640, bias=False)
                (to_v): LoRACompatibleLinear(in_features=640, out_features=640, bias=False)
                (to_out): ModuleList(
                  (0): LoRACompatibleLinear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): LoRACompatibleLinear(in_features=640, out_features=640, bias=False)
                (to_k): LoRACompatibleLinear(in_features=768, out_features=640, bias=False)
                (to_v): LoRACompatibleLinear(in_features=768, out_features=640, bias=False)
                (to_out): ModuleList(
                  (0): LoRACompatibleLinear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): LoRACompatibleLinear(in_features=640, out_features=5120, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): LoRACompatibleLinear(in_features=2560, out_features=640, bias=True)
                )
              )
            )
          )
          (proj_out): LoRACompatibleConv(640, 640, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
          (conv1): LoRACompatibleConv(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): LoRACompatibleLinear(in_features=1280, out_features=640, bias=True)
          (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): LoRACompatibleConv(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): LoRACompatibleConv(320, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
          (conv1): LoRACompatibleConv(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): LoRACompatibleLinear(in_features=1280, out_features=640, bias=True)
          (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): LoRACompatibleConv(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): LoRACompatibleConv(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (2): CrossAttnDownBlock2D(
      (attentions): ModuleList(
        (0-1): 2 x Transformer2DModel(
          (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
          (proj_in): LoRACompatibleConv(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=False)
                (to_k): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=False)
                (to_v): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=False)
                (to_out): ModuleList(
                  (0): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=False)
                (to_k): LoRACompatibleLinear(in_features=768, out_features=1280, bias=False)
                (to_v): LoRACompatibleLinear(in_features=768, out_features=1280, bias=False)
                (to_out): ModuleList(
                  (0): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): LoRACompatibleLinear(in_features=1280, out_features=10240, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): LoRACompatibleLinear(in_features=5120, out_features=1280, bias=True)
                )
              )
            )
          )
          (proj_out): LoRACompatibleConv(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
          (conv1): LoRACompatibleConv(640, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): LoRACompatibleConv(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): LoRACompatibleConv(640, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (conv1): LoRACompatibleConv(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): LoRACompatibleConv(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): LoRACompatibleConv(1280, 1280, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (3): DownBlock2D(
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (conv1): LoRACompatibleConv(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): LoRACompatibleConv(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
    )
  )
  (up_blocks): ModuleList(
    (0): UpBlock2D(
      (resnets): ModuleList(
        (0-2): 3 x ResnetBlock2D(
          (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
          (conv1): LoRACompatibleConv(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): LoRACompatibleConv(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): LoRACompatibleConv(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): LoRACompatibleConv(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (1): CrossAttnUpBlock2D(
      (attentions): ModuleList(
        (0-2): 3 x Transformer2DModel(
          (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
          (proj_in): LoRACompatibleConv(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=False)
                (to_k): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=False)
                (to_v): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=False)
                (to_out): ModuleList(
                  (0): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=False)
                (to_k): LoRACompatibleLinear(in_features=768, out_features=1280, bias=False)
                (to_v): LoRACompatibleLinear(in_features=768, out_features=1280, bias=False)
                (to_out): ModuleList(
                  (0): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): LoRACompatibleLinear(in_features=1280, out_features=10240, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): LoRACompatibleLinear(in_features=5120, out_features=1280, bias=True)
                )
              )
            )
          )
          (proj_out): LoRACompatibleConv(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
          (conv1): LoRACompatibleConv(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): LoRACompatibleConv(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): LoRACompatibleConv(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): ResnetBlock2D(
          (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
          (conv1): LoRACompatibleConv(1920, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): LoRACompatibleConv(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): LoRACompatibleConv(1920, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): LoRACompatibleConv(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (2): CrossAttnUpBlock2D(
      (attentions): ModuleList(
        (0-2): 3 x Transformer2DModel(
          (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
          (proj_in): LoRACompatibleConv(640, 640, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): LoRACompatibleLinear(in_features=640, out_features=640, bias=False)
                (to_k): LoRACompatibleLinear(in_features=640, out_features=640, bias=False)
                (to_v): LoRACompatibleLinear(in_features=640, out_features=640, bias=False)
                (to_out): ModuleList(
                  (0): LoRACompatibleLinear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): LoRACompatibleLinear(in_features=640, out_features=640, bias=False)
                (to_k): LoRACompatibleLinear(in_features=768, out_features=640, bias=False)
                (to_v): LoRACompatibleLinear(in_features=768, out_features=640, bias=False)
                (to_out): ModuleList(
                  (0): LoRACompatibleLinear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): LoRACompatibleLinear(in_features=640, out_features=5120, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): LoRACompatibleLinear(in_features=2560, out_features=640, bias=True)
                )
              )
            )
          )
          (proj_out): LoRACompatibleConv(640, 640, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
          (conv1): LoRACompatibleConv(1920, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): LoRACompatibleLinear(in_features=1280, out_features=640, bias=True)
          (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): LoRACompatibleConv(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): LoRACompatibleConv(1920, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (conv1): LoRACompatibleConv(1280, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): LoRACompatibleLinear(in_features=1280, out_features=640, bias=True)
          (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): LoRACompatibleConv(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): LoRACompatibleConv(1280, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): ResnetBlock2D(
          (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
          (conv1): LoRACompatibleConv(960, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): LoRACompatibleLinear(in_features=1280, out_features=640, bias=True)
          (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): LoRACompatibleConv(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): LoRACompatibleConv(960, 640, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): LoRACompatibleConv(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (3): CrossAttnUpBlock2D(
      (attentions): ModuleList(
        (0-2): 3 x Transformer2DModel(
          (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
          (proj_in): LoRACompatibleConv(320, 320, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): LoRACompatibleLinear(in_features=320, out_features=320, bias=False)
                (to_k): LoRACompatibleLinear(in_features=320, out_features=320, bias=False)
                (to_v): LoRACompatibleLinear(in_features=320, out_features=320, bias=False)
                (to_out): ModuleList(
                  (0): LoRACompatibleLinear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): LoRACompatibleLinear(in_features=320, out_features=320, bias=False)
                (to_k): LoRACompatibleLinear(in_features=768, out_features=320, bias=False)
                (to_v): LoRACompatibleLinear(in_features=768, out_features=320, bias=False)
                (to_out): ModuleList(
                  (0): LoRACompatibleLinear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): LoRACompatibleLinear(in_features=320, out_features=2560, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): LoRACompatibleLinear(in_features=1280, out_features=320, bias=True)
                )
              )
            )
          )
          (proj_out): LoRACompatibleConv(320, 320, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
          (conv1): LoRACompatibleConv(960, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): LoRACompatibleLinear(in_features=1280, out_features=320, bias=True)
          (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): LoRACompatibleConv(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): LoRACompatibleConv(960, 320, kernel_size=(1, 1), stride=(1, 1))
        )
        (1-2): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
          (conv1): LoRACompatibleConv(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): LoRACompatibleLinear(in_features=1280, out_features=320, bias=True)
          (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): LoRACompatibleConv(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): LoRACompatibleConv(640, 320, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
  )
  (mid_block): UNetMidBlock2DCrossAttn(
    (attentions): ModuleList(
      (0): Transformer2DModel(
        (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
        (proj_in): LoRACompatibleConv(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=False)
              (to_k): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=False)
              (to_v): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=False)
              (to_out): ModuleList(
                (0): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=False)
              (to_k): LoRACompatibleLinear(in_features=768, out_features=1280, bias=False)
              (to_v): LoRACompatibleLinear(in_features=768, out_features=1280, bias=False)
              (to_out): ModuleList(
                (0): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): LoRACompatibleLinear(in_features=1280, out_features=10240, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): LoRACompatibleLinear(in_features=5120, out_features=1280, bias=True)
              )
            )
          )
        )
        (proj_out): LoRACompatibleConv(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (resnets): ModuleList(
      (0-1): 2 x ResnetBlock2D(
        (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv1): LoRACompatibleConv(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): LoRACompatibleLinear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): LoRACompatibleConv(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
      )
    )
  )
  (conv_norm_out): None
  (conv_act): SiLU()
)

```



### Unet down_block的构造：

主函数是get_down_block,get_down_block函数主要作用是根据down_block_type类型生成下采样层，下采样层的类型主要有DownBlock2D和CrossAttnDownBlock2D两种，该函数的整体参数及含义如下：

```
down_block_type: str, #下采样块的类型
num_layers: int, #下采样块的层数
in_channels: int, #输入通道数
out_channels: int, #输出通道数
temb_channels: int, #时间嵌入通道数（其实就是时间嵌入维度）
add_downsample: bool, #是否添加下采样
resnet_eps: float, #ResNet模块的epsilon值
resnet_act_fn: str, #ResNet模块的激活函数
transformer_layers_per_block: int = 1, #每个块中的Transformer层数
num_attention_heads: Optional[int] = None, # 注意力头的数量
resnet_groups: Optional[int] = None, #ResNet模块的组数
cross_attention_dim: Optional[int] = None, #交叉注意力的维度
downsample_padding: Optional[int] = None, #下采样填充
dual_cross_attention: bool = False, #是否使用双交叉注意力
use_linear_projection: bool = False, #是否使用线性投影
only_cross_attention: bool = False, #是否仅使用交叉注意力
upcast_attention: bool = False, #是否使用向上转换的注意力
resnet_time_scale_shift: str = "default", #ResNet模块的时间尺度偏移
attention_type: str = "default", #注意力类型
resnet_skip_time_act: bool = False, #是否跳过时间激活
resnet_out_scale_factor: float = 1.0, #ResNet模块的输出缩放因子
cross_attention_norm: Optional[str] = None, #交叉注意力机制的规范化类型
attention_head_dim: Optional[int] = None, #注意力头的维度
downsample_type: Optional[str] = None, #下采样类型
dropout: float = 0.0, #丢弃概率
```



#### DownBlock2D：

首先由num_layers个ResnetBlock2D构成，接着根据是否需要添加下采样模块来确定是否需要DownSample2D模块

##### ResNetBlock2D：

在ResNetBlock2D中，可与图像融合的信息是时间信息，具体融合的方式是在图像编码归一化的时候融入时间信息

##### DownSample2D:

在下采样的过程中不涉及到时间信息以及其他信息与图像的融合，在DownSample2D中，不管使用什么样的下采样方式（卷积，平均池化等），都是把图片长宽的尺寸变为原来的1/2（如64* 64变成32* 32）

##### forward函数：

在forward函数中涉及到torch.utils.checkpoint.checkpoint()，简要的看一下这个函数，torch.utils.checkpoint.checkpoint()所需要的参数为(model,input1,input2,……) 其实就是拿到模型以及模型forward所需要的数据做前向传播



forward函数的内在逻辑很简单，输入图像编码以及时间编码，将各层ResNet块和下采样块（如果有的话）的结果记录到一个元组中，返回最终层的输出结果以及各层结果的元组



#### CrossAttnDownBlock2D:



根据是否采用双重注意力交叉机制有两种Transformer块：Transformer2DModel和DualTransformer2DModel

##### Transformer2DModel：

Transformer2DModel由input_layers+Transformer_Block+output_layers(仍需补充)构成

* input_layers其实就是做组归一化+投影/1*1的卷积

* Transformer_Block由k个BasicTransformerBlock块构成
* out_putlayers也是投影/1*1卷积



###### BasicTransformerBlock:

BasicTransformerBlock其实是 自注意力+控制信息融合（这里不确定）+交叉注意力+前馈层+PixArt-Alpha blocks（这个块应该是用于文生图的，可选）的形式，其中除了在控制信息融合之后没有用残差连接，在其他情况下都用了残差连接。先明确一下在模块中涉及到的重要的参数：hidden_states其实是query，encoder_hidden_states是key/value，看一下具体流程：

1. 自注意力

   * 先对hidden_states做归一化得到norm_hidden_states，
   * 接着，对norm_hidden_states做位置编码
   * hidden_states做自注意力得到attn_output
   * 将attn_output与hidden_states做残差连接(attn_output+hidden_states)得到新hidden_states

2. 控制信息融合

   * 代码中是GLIGEN control(这里还要再研究一下)，将控制信息与hidden_states融合得到新hidden_states

3. 交叉注意力

   交叉注意力部分的1 2 4步骤与自注意力是一样的，只是在第3步的时候是将hidden_states（query）与encoder_hidden_states（key/value）放在一起做交叉注意力得到新的hidden_states

4. 前馈层

   将hidden_states放入前馈层得到ff_out，将ff_out与hidden_states做残差连接后得到最终的hidden_states

5. 将hidden_states的结果作为最终结果输出

上面的过程可以用下图表示：

![v2-2797c76152e6aa44edbaa94af0a41c28_r](C:\Users\Shipu\Desktop\v2-2797c76152e6aa44edbaa94af0a41c28_r.jpg)

