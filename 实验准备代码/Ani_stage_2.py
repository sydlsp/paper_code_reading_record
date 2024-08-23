import argparse
import copy # 提供浅拷贝和深拷贝操作
import logging # 日志记录
import math
import os.path
import pdb
import os.path as osp
import random
import time
import warnings
from collections import OrderedDict # 字典子类，记住了插入键值对的顺序
from datetime import datetime
from pathlib import Path # 以面向对象的方式来处理文件和目录路径
from tempfile import TemporaryDirectory # TemporaryDirectory类用于创建一个临时目录，当退出with语句时，临时目录会被自动删除

import diffusers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs # 用于配置分布式数据并行DDP的参数
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from omegaconf import OmegaConf # 用于处理配置文件和配置对象
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection

# 导入自己的代码
from ani_src.dataset.dataset_face import FaceDataset,FaceDatasetValid,collate_fn
from ani_src.models.mutual_self_attention import ReferenceAttentionControl
from ani_src.models.pose_guider import PoseGuider
from ani_src.models.unet_2d_condition import UNet2DConditionModel
from ani_src.models.unet_3d import UNet3DConditionModel
from ani_src.pipelines.pipeline_pose2vid import Pose2VideoPipeline
from ani_src.utils.util import (
    delete_additional_ckpt,
    import_filename,
    read_frames,
    save_videos_grid,
    seed_everything,
)

"""
前期的准备工作
"""

# 忽略所有警告信息
warnings.filterwarnings("ignore")

# 检查用户安装的diffusers库版本是否低于满足的最低版本要求
check_min_version("0.10.0.dev0")

# accelerate.logging.get_logger()函数用于获取一个logger对象，用于记录日志信息
logger=get_logger(__name__,log_level="INFO")

"""
Net类
"""
class Net(nn.Module):
    def __init__(
            self,
            reference_unet:UNet2DConditionModel, # 提供类型注解
            denoising_unet:UNet3DConditionModel,
            pose_guider:PoseGuider,
            reference_control_writer,
            reference_control_reader,
    ):
        super().__init__()
        self.reference_unet=reference_unet
        self.denoising_unet=denoising_unet
        self.pose_guider=pose_guider
        self.reference_control_writer=reference_control_writer
        self.reference_control_reader=reference_control_reader

    def forward(
            self,
            noisy_latents,
            time_steps,
            ref_image_latents,
            clip_image_embeds,
            pose_image,
            ref_pose_image,
            uncond_fwd:bool=False,
    ):

        pose_cond_tensor=pose_image.to(device="cuda")  # reference_pose_image
        ref_pose_tensor=ref_pose_image.to(device="cuda") # target_pose_images

        # 对应论文中Reference Pose Image和Target Pose Images一起被放入PoseGuider中
        pose_fea=self.pose_guider(pose_cond_tensor,ref_pose_tensor)

        # 如果进行的是有条件的前向传播
        if not uncond_fwd:
            ref_timesteps=torch.zeros_like(time_steps)
            self.reference_unet(
                ref_image_latents,  # 加噪完的图像
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds, # 条件就是clip编码
                return_dict=False,
                )
            # reference_net和denoising_net的交互
            self.reference_control_reader.update(self.reference_control_writer)
        else:
            pass

        # 也就是说在这里，“条件”仍然是clip编码
        # pose_cond_fea是在下采样的过程中不断注入的
        model_pred=self.denoising_unet(
            noisy_latents,
            time_steps,
            pose_cond_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred

"""
信噪比计算函数
"""
def compute_snr(noise_scheduler,timesteps):
    alphas_cumprod=noise_scheduler.alphas_cumprod  # alpha值的累乘
    sqrt_alphas_cumprod=alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod=(1.0-alphas_cumprod)**0.5

    # 根据timesteps选出对应的alpha的累乘值
    sqrt_alphas_cumprod=sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    # 这种while写法来扩充维度的做法值得学习一下
    while len(sqrt_alphas_cumprod.shape)<len(timesteps.shape):
        sqrt_alphas_cumprod=sqrt_alphas_cumprod[...,None]
    # 扩充完维度后填补内容
    alpha=sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape)<len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod[...,None]
    sigma=sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)
    """
    在这里我们所说的信噪比其实就是加噪公式中
    x_t=alpha*x_0+bata*N(0,1) 根据timestep抽出对应的alpha和beta然后按照下面的公式计算信噪比
    """
    snr=(alpha/sigma)**2
    return snr

"""
评测函数
"""
def log_validation(
        vae,
        image_enc,
        net,
        scheduler,
        accelerator,
        width,
        height,
        clip_length=24,
        generator=None,
        valid_dataset=None,
):
    logger.info("Running validation...")
    # 从实例中提取出原始的模型对象
    ori_net=accelerator.unwrap_model(net)
    reference_unet=ori_net.reference_unet
    denoising_unet=ori_net.denoising_unet
    pose_guider=ori_net.pose_guider

    if generator is None:
        generator=torch.manual_seed(42)
    tmp_denoising_unet=copy.deepcopy(denoising_unet)
    tmp_denoising_unet=tmp_denoising_unet.to(dtype=torch.float16)

    pipe=Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )

    pipe=pipe.to(accelerator.device)

    dataset_len=len(valid_dataset)
    sample_idx=[random.randint(0,dataset_len) for _ in range(2)]

    results=[]
    for idx in sample_idx:
        sample=valid_dataset[idx]

        ref_image_pil = Image.fromarray(sample['ref_img']).convert("RGB")
        pose_images = [Image.fromarray(sample['pixel_values_pose'][idx]).convert("RGB") for idx in
                       range(sample['pixel_values_pose'].shape[0])]
        gt_images = [Image.fromarray(sample['tar_gt'][idx]).convert("RGB") for idx in range(sample['tar_gt'].shape[0])]

        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )

        pose_tensor_list = []
        ref_tensor_list = []
        gt_tensor_list = []
        pose_list = []

        for pose_image_pil in pose_images[:clip_length]:
            pose_tensor_list.append(pose_transform(pose_image_pil))
            ref_tensor_list.append(pose_transform(ref_image_pil))
        for gt_image_pil in gt_images[:clip_length]:
            gt_tensor_list.append(pose_transform(gt_image_pil))

        pose_list = sample['pixel_values_pose'][:clip_length]
        ref_pose = sample['pixel_values_ref_pose']

        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)  # (c, f, h, w)

        ref_tensor = torch.stack(ref_tensor_list, dim=0)  # (f, c, h, w)
        ref_tensor = ref_tensor.transpose(0, 1)  # (c, f, h, w)

        gt_tensor = torch.stack(gt_tensor_list, dim=0)  # (f, c, h, w)
        gt_tensor = gt_tensor.transpose(0, 1)  # (c, f, h, w)

        pipeline_output = pipe(
            ref_image_pil,
            pose_list,
            ref_pose,
            width,
            height,
            clip_length,
            25,
            3.5,
            generator=generator,
        )
        video = pipeline_output.videos

        # Concat it with pose tensor
        pose_tensor = pose_tensor.unsqueeze(0)
        ref_tensor = ref_tensor.unsqueeze(0)
        gt_tensor = gt_tensor.unsqueeze(0)
        video = torch.cat([ref_tensor, pose_tensor, video, gt_tensor], dim=0)

        results.append({"name": f"sample_{idx}", "vid": video})

    del tmp_denoising_unet
    del pipe
    torch.cuda.empty_cache()

    return results

"""
主函数
"""
def main(cfg):
    # 配置分布式数据并行DDP的参数
    kwargs=DistributedDataParallelKwargs(find_unused_parameters=False)

    # 创建Accelerator实例，用来处理分布式训练
    accelerator=Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps, # 梯度累积步数
        mixed_precision=cfg.solver.mixed_precision, # 混合精度训练,'fp16'
        kwargs_handlers=[kwargs], # 分布式数据并行DDP的参数
    )

    # 用logging模块来配置基本的日志设置
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", # 日志格式
        datefmt="%m/%d/%Y %H:%M:%S", # 日期格式
        level=logging.INFO, # 日志级别，只有INFO级别及以上的日志信息才会被记录
    )

    # 这个logger是我们实例化出来的logger对象，用于记录日志信息
    # main_process_only=False表示在所有进程中都记录日志信息
    logger.info(accelerator.state,main_process_only=False)

    # 设置transformer和diffusers库的日志级别
    if accelerator.is_local_main_process:
        # 将日志级别设置为waring
        transformers.utils.logging.set_verbosity_waring()
        # 将日志级别设置为info
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 设置随机种子
    if cfg.seed is not None:
        # 这里调用seed_everything()函数设置随机种子
        seed_everything(cfg.seed)

    # 读取实验名称
    exp_name=cfg.exp_name

    # 将实验名称和输出目录拼接成保存目录
    save_dir=f"{cfg.output_dir}/{exp_name}"

    # 在主进程中创建保存目录
    if accelerator.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # 创建采样目录
    sample_dir=os.path.join(save_dir,"samples")

    if accelerator.is_main_process and not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # 推理的配置文件路径
    inference_config_path="./configs/inference/inference_v2.yaml"
    infer_config=OmegaConf.load(inference_config_path)

    if cfg.weight_dtype=="float16":
        weight_dtype=torch.float16
    elif cfg.weight_dtype=="float32":
        weight_dtype=torch.float32
    else:
        raise ValueError(
            f" Unsupported weight_dtype: {cfg.weight_dtype} during training"
        )

    # 将noise_sheduler配置转化为字典
    sched_kwargs=OmegaConf.to_container(cfg.noise_scheduler_kwargs)

    # 如果启用零信噪比模式，那么需要对噪声调度器的参数配置进行更新
    # 下面的方法其实就是补充新的键值对
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )

    # 验证期间的噪声调度器
    val_noise_scheduler=DDIMScheduler(**sched_kwargs)

    # 训练期间的噪声调度器
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler=DDIMScheduler(**sched_kwargs)

    # 图像编码器
    image_enc=CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path
    ).to(dtype=weight_dtype,device="cuda")

    # vae编码器
    vae=AutoencoderKL.from_pretrained(cfg.vae_model_path).to("cuda",dtype=weight_dtype)

    # reference_unet是2D条件模型
    reference_unet=UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet"
    ).to(dtype=weight_dtype,device="cuda")

    # 去噪Unet是一个3D条件模型
    # 在这里，motion_module暂时理解为一个处理时序3D数据的组件，可以增强模型在处理具有时间维度的数据时的表现
    denoising_unet=UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        cfg.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),
    ).to(dtype=weight_dtype,device="cuda")

    # PoseGuider网络
    pose_guider=PoseGuider(noise_latent_channels=320).to(device="cuda",dtype=weight_dtype)

    stage1_ckpt_dir=cfg.stage1_ckpt_dir
    stage1_ckpt_step=cfg.stage1_ckpt_step

    # 加载模型权重

    # map_location="cpu"表示将模型加载到CPU上
    # strict=False表示不严格加载模型权重
    denoising_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir,f"denoising_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),strict=False,)

    reference_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"reference_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    pose_guider.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"pose_guider-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )

    # 将模型权重全部冻结起来
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    pose_guider.requires_grad_(False)

    # 现在，我们仅设置denoising_unet中的motion module为可训练(也就是说仅仅让时序性的模块可训练)
    for name,module in denoising_unet.named_modules():
        if "motion_module" in name:
            for params in module.parameters():
                params.requires_grad=True

    # 下面两个类的实例的主要功能是修改reference_unet模块的自注意力和组归一化来实现
    # 我们来关注一下内部的大致原理:其基本原理就是有一个bank来存储unet中的注意力特征
    # writer将特征写入自己的bank中，read通过调用update方法读取writer中存储的特征
    reference_control_writer=ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )

    reference_control_reader=ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    # 到这里，我们需要的模块的实例化都完成了，需要将其拼成一个总的模型
    # 也就是说，我们要实现一个Net类

    # 实例化Net类
    net=Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,)

    # 判断是否要在reference_unet和denosing_unet中使用内存高效的注意力机制
    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not installed, please install it to use memory efficient attention"
            )

    # 判断是否使用梯度检查点
    if cfg.solver.enable_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    # 根据配置文件来动态调整学习率
    if cfg.solver.scale_lr:
        learning_rate=(cfg.solver.learning_rate
                       *cfg.solver.gradient_accumulation_steps
                       *cfg.train_bs
                       *accelerator.num_processes)
    else:
        learning_rate=cfg.solver.learning_rate

    # 配置优化器
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam optimizer"
            )

        optimizer_cls=bnb.optim.AdamW8bit
    else:
        optimizer_cls=torch.optim.AdamW

    # 获取网络中所有可学习的参数
    # p是遍历net.parameters()，filter选择出需要梯度的参数
    trainable_params=list(filter(lambda p:p.requires_grad,net.parameters()))
    logger.info(f"Total trainable parameters {len(trainable_params)}")

    optimizer=optimizer_cls(
        trainable_params,
        lr=learning_rate,
        batas=(cfg.solver.adam_beta1,cfg.solver.adam_beta2), # 优化器的beta参数
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # 学习率调度器
    lr_scheduler=get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps* cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps* cfg.solver.gradient_accumulation_steps,)

    # 读数据
    train_dataset=FaceDataset(**cfg.data,is_image=False)
    valid_dataset=FaceDatasetValid(**cfg.data,is_image=False)

    # 创建dataloader
    train_dataloader=torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train_bs,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        collate_fn=collate_fn,)

    # 用accelerator.prepare准备好要用到的东西
    (net,optimizer,train_dataloader,lr_scheduler)=accelerator.prepare(net,optimizer,train_dataloader,lr_scheduler)

    # 计算每个epoch的更新步数，采用梯度累积可以减少现存的使用
    num_update_step_per_epoch=math.ceil(
        len(train_dataloader)/cfg.solver.gradient_accumulation_steps
    )

    # 计算总的训练轮数
    num_train_epochs=math.ceil(
        cfg.solver.max_train_steps/num_update_step_per_epoch
    )

    if accelerator.is_main_process:
        run_time=datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(exp_name,)


    """
    开始训练
    """

    # 总的batch_size
    total_batch_size=(cfg.train_bs*accelerator.num_processes*cfg.solver.gradient_accumulation_steps)

    # 在控制台输出部分训练信息
    logger.info("*** Training ***")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")

    # 这两个变量是为了下面恢复检查点而设置的
    global_step=0
    first_epoch=0

    # 恢复检查点
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint!="latest":
            resume_dir=cfg.resume_from_checkpoint
        else:
            resume_dir=save_dir

        dirs=os.listdir(resume_dir)
        # 从目录中找到以checkpoint开头的目录或者文件
        dirs=[d for d in dirs if d.startswith("checkpoint")]
        dirs=sorted(dirs,key=lambda x:int(x.split("-")[1]))
        path=dirs[-1]
        accelerator.load_state(os.path.join(resume_dir,path))
        accelerator.print(f"Resuming from checkpoint{path}")
        global_step=int(path.split("-")[1])

        first_epoch=global_step//num_update_step_per_epoch
        resume_step=global_step%num_update_step_per_epoch


    # 只在每台机器上显示一次进度条
    progress_bar=tqdm(range(global_step,cfg.solver.max_train_steps),disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # 正式开始训练
    for epoch in range(first_epoch,num_train_epochs):
        train_loss=0.0
        t_data_start=time.time()

        for step,batch in enumerate(train_dataloader):
            t_data=time.time()-t_data_start

            # 指定在net上进行梯度累积
            with accelerator.accumulate(net):
                # 将视频转移到潜在空间中
                # batch["pixel_values"] [batch_size,n_frames,channels,height,weight]
                pixel_values_vid=batch["pixel_values"].to(weight_dtype)
                with torch.no_grad():
                    video_length=pixel_values_vid.shape[1]

                    # 修改输入视频的形状
                    pixel_values_vid=rearrange(pixel_values_vid,"b f c h w -> (b f) c h w")

                    # 将视频映射到潜在空间中
                    latents=vae.encode(pixel_values_vid).latent_dist.sample()

                    latents=rearrange(latents,"(b f) c h w -> b c f h w",f=video_length)

                    latents=latents*0.18215

                # 生成原始噪声
                noise=torch.randn_like(latents)

                if cfg.noise_offset>0:
                    noise+=cfg.noise_offset*torch.randn((latents.shape[0],latents.shape[1],1,1,1),device=latents.device)

                bsz=latents.shape[0]

                # 为每个视频随机挑选一个时间步
                timesteps=torch.randint(0,train_noise_scheduler.num_train_steps,(bsz,),device=latents.device)

                timesteps=timesteps.long()

                # 姿势图
                pixel_values_pose=batch["pixel_values_pose"]  # [batch_size,frames,c,h,w]
                pixel_values_pose=pixel_values_pose.transpose(1,2) #[batch_size,c,frames,h,w]

                # pixel_values_ref_pose [batch_size,c,h,w]
                pixel_values_ref_pose=batch["pixel_values_ref_pose"] # ref_image 的姿势图

                # 随机选择是否进行无条件的前向传播
                uncond_fwd=random.random()<cfg.uncond_ratio
                clip_image_list=[]
                ref_image_list=[]

                for batch_idx,(ref_img,clip_img) in enumerate(
                    zip(
                        batch["pixel_values_ref_img"], # [batch_size,c,256,256]
                        batch["clip_ref_image"] # [batch_size,c,224,224] 相当于放进clip的版本的ref_image
                    )
                ):
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                    else:
                        clip_image_list.append(clip_img)
                    ref_image_list.append(ref_img)


                with torch.no_grad():
                    # 参考图像过vae是上面reference_net需要
                    ref_img=torch.stack(ref_image_list,dim=0).to(detype=vae.dtype,device=vae.device)

                    ref_image_latents=vae.encode(ref_img).latent_dist.sample()

                    ref_image_latents=ref_image_latents*0.18215

                    # 参考图像clip编码
                    clip_image=torch.stack(clip_image_list,dim=0).to(dtype=image_enc.dtype,device=image_enc.device)
                    clip_image=clip_image.to(device="cuda",dtype=weight_dtype)

                    clip_image_embeds=image_enc(clip_image.to("cuda"),dtype=weight_dtype).image_embeds
                    clip_image_embeds=clip_image_embeds.unsqueeze(1)


                # 在latents上加噪,也就是在视频帧上加噪声
                noisy_latents=train_noise_scheduler.add_noise(latents,noise,timesteps)

                # 根据噪声调度器的类型来确定损失函数的目标
                if train_noise_scheduler.prediction_type=="epsilon":
                    target=noise
                elif train_noise_scheduler.prediction_type=="v_prediction":
                    target=train_noise_scheduler.get_velocity(latents,noise,timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )

                """
                前向传播
                在前向传播这里我们再梳理一下：
                两个pose首先被放入PoseGuider中生成pose_fea
                在reference_unet中，输入是reference_image_latents(注意，这里是没有加噪往里放的)，条件是reference_img的clip编码，time_step是特殊的0(这里现在理解的话是由于reference_image_latents没有加噪)，所以timestep用的全部为0
                在denoising_unet中，输入是noisy_latents(视频帧加噪的结果)，条件是reference_image的clip编码，pose_fea在下采样的过程中逐渐加入模型中，timestep是noisy_latents加噪用的timestep
                """
                model_pred=net(noisy_latents,timesteps,ref_image_latents,clip_image_embeds,
                               pixel_values_pose,pixel_values_ref_pose,uncond_fwd=uncond_fwd)

                if cfg.snr_gamma==0:
                    """
                    在扩散模型中，snr_gamma可以用来平衡模型的去噪能力与生成能力，较高的snr_gamma意味着模型在
                    初期噪声较高的时候更加关注信号部分，较低的snr_gamma则可能使得模型在噪声较低的时候更关注信号部分
                    信噪比snr=P_signal/P_noise,以分贝为单位的话snr(db)=10*log10(P_signal/P_noise)
                    """
                    loss=F.mse_loss(
                        model_pred.float(),target.float(),reduction="mean"
                    )
                else:
                    # 这里涉及到计算信噪比了，我们需要构建一个计算信噪比的函数
                    snr=compute_snr(train_noise_scheduler,timesteps)
                    # 在这里预测模式是一个值得关注的问题，一共有三种预测模式epsilon,sample以及v_prediction
                    # 如果scheduler采用eplsion,这意味着扩散模型的输出其实是在预测噪声,采样器根据公式来计算sample(原始样本)
                    # 如果scheduler采用sample，这意味着扩散模型的输出其实是在预测sample(原始样本),采样器根据公式来计算noise(eplsion)
                    # 如果scheduler采用v_prediction，这种模式类似与上面两种方式的结合sample和eplsion都是根据公式来计算的
                    # v_prediction模式意味着扩散模型可能的输出是噪声方差或者是变异性的估计
                    if train_noise_scheduler.config.prediction_type=="v_prediction":
                        snr=snr+1

                    mse_loss_weights=(
                            torch.stack([snr,cfg.snr_gamma*torch.ones_like(timesteps)],dim=1).min(dim=1)[0]/snr
                    )

                    loss=F.mse_loss(
                        model_pred.float(),target.float(),reduction="none"
                    )
                    # 对loss张量的除了第一个维度进行平均，得到每个样本的平均损失,这里的结果其实就是将每个样本平均成一个数字
                    loss=(loss.mean(dim=list(range(1,len(loss.shape))))*mse_loss_weights)
                    loss=loss.mean()

                # 计算并记录分布式训练中的平均损失
                avg_loss=accelerator.gather(loss.repeat(cfg.train_bs)).mean()
                train_loss+=avg_loss.item()/cfg.solver.gradient_accumulation_steps

                # 反向传播
                accelerator.backward(loss)
                # 检查完同步梯度后
                if accelerator.sync_gradients:
                    # 梯度裁剪
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # 将bank中的注意力清除
                if accelerator.sync_gradients:
                    reference_control_reader.clear()
                    reference_control_writer.clear()

                    progress_bar.update(1)
                    global_step+=1
                    accelerator.log({"train_loss":train_loss},step=global_step)
                    train_loss=0.0

                    # 验证板块，如果当前总的训练步数满足条件，那么开始检查
                    if (global_step%cfg.val.validation_steps==0) or (global_step in cfg.val.validation_steps_tuple):
                        if accelerator.is_main_process:
                            generator=torch.Genrator(device=accelerator.device)
                            generator.manual_seed(cfg.seed)

                            # 在这里我们要设计评测函数log_validation,在评测函数中涉及到pose2video pipeline
                            # pipeline的输出结果包括视频
                            sample_dicts=log_validation(
                                vae=vae,
                                image_enc=image_enc,
                                net=net,
                                scheduler=val_noise_scheduler,
                                accelerator=accelerator,
                                width=cfg.data.sample_size[0],
                                height=cfg.data.sample_size[1],
                                clip_length=cfg.data.sample_n_frames,
                                generator=generator,
                                valid_dataset=valid_dataset
                            )

                            for sample_id,sample_dict in enumerate(sample_dicts):
                                sample_name=sample_dict["name"]
                                vid=sample_dict["vid"]
                                out_file=os.path.join(sample_dir,f'{global_step:06d}-{sample_name}.gif')
                                save_videos_grid(vid,out_file,n_rows=4)

                            reference_control_writer=ReferenceAttentionControl(
                                reference_unet,
                                do_classifier_free_guidance=False,
                                mode="write",
                                fusion_blocks="full",
                            )

                            reference_control_reader=ReferenceAttentionControl(
                                denoising_unet,
                                do_classifier_free_guidance=False,
                                mode="read",
                                fusion_blocks="full",
                            )

                logs={
                    "step_loss":loss.detach().item(),
                    "lr":lr_scheduler.get_last_lr()[0],
                    "td":f"{t_data:.2f}s",
                }

                t_data_start=time.time()
                # 在进度条的末尾显示logs中要展示的信息
                progress_bar.set_postfix(**logs)

                if global_step>=cfg.solver.max_train_steps:
                    break

            # 在每一轮训练完成后保存信息
            if accelerator.is_main_process:
                save_path=os.path.join(save_dir,f"checkpoint-{global_step}")
                delete_additional_ckpt(save_dir,1)
                accelerator.save_state(save_path)

                # 仅保存motion module
                unwrap_net=accelerator.unwrap_model(net)
                # 这里设计了新的函数来保存motion_module
                save_checkpoint(
                    unwrap_net.denoising_unet,
                    save_dir,
                    "motion_module",
                    global_step,
                    total_limit=3
                )

        accelerator.wait_for_everyone()
        accelerator.end_training()

def save_checkpoint(model,save_dir,prefix,ckpt_num,total_limit=None):
    # prefix就是
    save_path=osp.join(save_dir,f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints=os.listdir(save_dir)
        checkpoints=[d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    mm_state_dict = OrderedDict()
    state_dict=model.state_dict()

    for key in state_dict:
        if "motion_module" in key:
            mm_state_dict[key]=state_dict[key]

    torch.save(mm_state_dict,save_path)











































