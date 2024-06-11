from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

# 加载数据集
dataset=load_dataset(path="imagefolder",data_dir="/share/test/shipu/condition_model/face_data")

# 区分训练集、验证集和测试集
train_dataset=dataset["train"]
val_dataset=dataset["validation"]
test_dataset=dataset["test"]

# 定义数据增强过程
from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])

    ]
)

# 定义将数据增强应用到数据集的方法

def transform(examples):
    images=[preprocess(image.convert("RGB")) for image in examples['image']]
    return {"image":images}

# 在对应数据集中应用数据增强方法
train_dataset.set_transform(transform)
val_dataset.set_transform(transform)

# 数据增强效果展示
print(val_dataset['image'][0].shape)

# 将数据划分为一个一个batch
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

# 加载数据集
dataset=load_dataset(path="imagefolder",data_dir="/share/test/shipu/condition_model/face_data")

# 区分训练集、验证集和测试集
train_dataset=dataset["train"]
val_dataset=dataset["validation"]
test_dataset=dataset["test"]

# 定义数据增强过程
from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])

    ]
)

# 定义将数据增强应用到数据集的方法

def transform(examples):
    images=[preprocess(image.convert("RGB")) for image in examples['image']]
    return {"image":images}

# 在对应数据集中应用数据增强方法
train_dataset.set_transform(transform)
val_dataset.set_transform(transform)

# 数据增强效果展示
print(val_dataset['image'][0].shape)

# 将数据划分为一个一个batch
import torch
batch_size=4
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

# 定义模型
import os
from torch import nn
import torchvision
from torchvision.models import resnet50

from diffusers import AutoencoderKL, UNet2DConditionModel

# 修改预训练权重的下载地址不在C盘中
os.environ['TORCH_HOME'] = "/share/ckpt/shipu/local_model_weight/Resnet50"


class Test_Model(nn.Module):
    def __init__(self, vae_save_dir, condition_unet_save_dir, ):
        super().__init__()

        self.model_1 = resnet50(pretrained=True)

        self.model_2 = resnet50(pretrained=True)

        self.model_3 = resnet50(pretrained=True)

        self.model_1.fc = nn.Linear(2048, 256, bias=True)

        self.model_2.fc = nn.Linear(2048, 256, bias=True)

        self.model_3.fc = nn.Linear(2048, 256, bias=True)

        self.vae = AutoencoderKL.from_pretrained(vae_save_dir)

        self.unet = UNet2DConditionModel.from_pretrained(condition_unet_save_dir)

    # 在这里要想明白unet2dconditionmodel模型需要什么，模型要的是 对干净图像加完噪声的高斯噪声图、条件、time_step
    # 先想给干净的图加噪需要准备什么：
    #    首先要准备干净的图，这里是latents，还要准备高斯噪声noise以及timestep
    #    为了能将噪声加到干净的图上，需要有一个noise_scheduler
    # 再想time_step和什么是相关的：
    #     1. 给干净图像加噪时需要time_step
    #     2. unet2Dconditionmodel去噪需要time_step
    def forward(self, x, time_steps, noise, noise_scheduler):
        batch_size = x.shape[0]

        # time_steps=torch.randint(0,1000,(batch_size,),device=x.device(),dtype=torch.int64)

        # 要latents是做什么的，要想明白了，要latents其实是干净图像，是为了加噪的
        latents = self.vae.encode(x).latent_dist.sample()

        # 用noise_scheduler来给干净的图片加噪

        noisy_image = noise_scheduler.add_noise(latents, noise, time_steps)

        # 条件就是三个东西拼起来,中间升一个维度是因为unetconditionmodel的条件形状要要求[batch_size,len_squence,token_embedding_size]
        encoder_hidden_states = torch.cat((self.model_1(x), self.model_2(x), self.model_3(x)), dim=-1).unsqueeze(dim=1)

        # 现在已经有 noisy_image、条件以及timestep了
        noise_pred = self.unet(sample=noisy_image, timestep=time_steps,
                               encoder_hidden_states=encoder_hidden_states).sample

        return noise_pred

# 定义训练过程：想明白了在训练的时候需要什么，首先需要模型，数据，训练轮数，noise_scheduler，优化器，以及学习率调度器

from accelerate import Accelerator
from tqdm.auto import tqdm
import torch.nn.functional as F



def train(model, train_dataloader, epoches, noise_scheduler, optimizer, lr_scheduler):
    output_dir = "./condition_2d_model"

    # 实例化accelerate类
    # accelerator = Accelerator(
    #     mixed_precision="fp16",
    #     gradient_accumulation_steps=1,
    #
    #     # 日志记录的方式
    #     log_with="tensorboard",
    #
    #     # 指定日志文件的存储目录
    #     project_dir=os.path.join(output_dir, "logs")
    # )

    # 确保只有主进程执行下面的操作
    # if accelerator.is_main_process:
    #
    #     if output_dir is not None:
    #         os.makedirs(output_dir, exist_ok=True)
    #
    #     accelerator.init_trackers("train_example")

    # 包装

    # model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader,lr_scheduler)

    global_step = 0
    print("model------------------------------------------------")
    # print(model.modules().device)

    for epoch in range(epoches):

        print("len",len(train_dataloader))
        # progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar = tqdm(total=len(train_dataloader))
        print("here-------------------------------------------")
        progress_bar.set_description(f"Epoch{epoch}")

        print("0------------------")
        for step, batch in enumerate(train_dataloader):
            x = batch["image"].cuda()

            print("1--------------")

            # 创建噪声
            noise = torch.randn(size=[x.shape[0], 4, x.shape[2] // 8, x.shape[3] // 8], device=x.device)

            # 创建timesteps
            time_steps = torch.randint(0, 1000, (x.shape[0],), device=x.device)
            print("2-------------------------------")
            # with accelerator.accumulate(model):
            noisy_pred = model(x, time_steps, noise, noise_scheduler)
            loss = F.mse_loss(noise, noisy_pred)

            loss.backward()
            # accelerator.backward(loss)
            #
            # accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": "%.8f" % lr_scheduler.get_last_lr()[0], "step": global_step}
            print(type(lr_scheduler.get_last_lr()[0]))
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)
            global_step += 1


# 实例化模型
vae_save_dir,condition_unet_save_dir="/share/ckpt/shipu/local_model_weight/vae","/share/ckpt/shipu/local_model_weight/conditional_unet"
model=Test_Model(vae_save_dir=vae_save_dir,condition_unet_save_dir=condition_unet_save_dir)

model=model.cuda()
device_ids=[0,1]
model=nn.DataParallel(model,device_ids=device_ids)

# 下面来实例化noise_scheduler，优化器，以及学习率调度器
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup


noise_scheduler=DDPMScheduler(num_train_timesteps=1000)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
lr_scheduler=get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=30,
    num_training_steps=(len(train_dataloader)*batch_size)
)

# 看一下输入网络的数据是否正确
# 将准备好的干净的数据,time_step,noise,noise_scheduler放入网络中
for batch in train_dataloader:
    data_in=batch["image"]
    print(data_in.shape)
    time_step_in=torch.randint(0,1000,(batch_size,),device=data_in.device)
    noise_in=torch.randn(size=[batch_size,4,data_in.shape[2]//8,data_in.shape[3]//8],device=data_in.device)
    noise_pred=model(x=data_in,time_steps=time_step_in,noise=noise_in,noise_scheduler=noise_scheduler)
    print(noise_pred.shape)
    break

from accelerate import notebook_launcher

# epoches=2
# args=(model, train_dataloader, epoches, noise_scheduler, optimizer, lr_scheduler)
#
# notebook_launcher(train,args,num_processes=2)

epoches=300
train(model, train_dataloader, epoches, noise_scheduler, optimizer, lr_scheduler)
torch.save(model.state_dict(),"/share/ckpt/shipu/local_model_weight/train_weight/save_weight_300.pt")