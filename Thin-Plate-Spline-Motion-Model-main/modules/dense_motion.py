from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian
from modules.util import to_homogeneous, from_homogeneous, UpBlock2d, TPS
import math

"""
DenseMotionNetWork模块从k个TPS变换和一个仿射变换估计光流，根据沙漏网络输出多分辨率遮挡掩码
"""
class DenseMotionNetwork(nn.Module):
    """
    Module that estimating an optical flow and multi-resolution occlusion masks 
                        from K TPS transformations and an affine transformation.
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_tps, num_channels, 
                 scale_factor=0.25, bg = False, multi_mask = True, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()

        if scale_factor != 1:
            #下采样方法，下采样其实减小了图片大小，这里用的是一种抗锯齿的下采样方法
            self.down = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.scale_factor = scale_factor
        self.multi_mask = multi_mask

        #创建沙漏网络，沙漏网络常用于估计人体姿态
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_channels * (num_tps+1) + num_tps*5+1),
                                   max_features=max_features, num_blocks=num_blocks)

        #这个输出的结果是decoder各个卷积层的输入维度,最后一个数字是block_expansion + in_features
        hourglass_output_size = self.hourglass.out_channels

        #再是一个卷积层将沙漏网络的结果做个卷积
        self.maps = nn.Conv2d(hourglass_output_size[-1], num_tps + 1, kernel_size=(7, 7), padding=(3, 3))

        #使用多重掩码的情况
        if multi_mask:
            up = []

            #根据缩放因子计算上采样的次数
            self.up_nums = int(math.log(1/scale_factor, 2))

            #遮挡数量
            self.occlusion_num = 4

            #channel列表包含每个上采样步骤的输出通道数
            channel = [hourglass_output_size[-1]//(2**i) for i in range(self.up_nums)]

            #对于每个上采样次数，创建一个上采样模块，并将其转化为模型序列
            for i in range(self.up_nums):
                up.append(UpBlock2d(channel[i], channel[i]//2, kernel_size=3, padding=1))
            self.up = nn.ModuleList(up)

            #此时，channel包含每个遮挡模块的输入通道数
            channel = [hourglass_output_size[-i-1] for i in range(self.occlusion_num-self.up_nums)[::-1]]

            for i in range(self.up_nums):
                channel.append(hourglass_output_size[-1]//(2**(i+1)))
            occlusion = []
            
            for i in range(self.occlusion_num):
                occlusion.append(nn.Conv2d(channel[i], 1, kernel_size=(7, 7), padding=(3, 3)))
            self.occlusion = nn.ModuleList(occlusion)
        else:
            occlusion = [nn.Conv2d(hourglass_output_size[-1], 1, kernel_size=(7, 7), padding=(3, 3))]
            self.occlusion = nn.ModuleList(occlusion)

        self.num_tps = num_tps
        self.bg = bg
        self.kp_variance = kp_variance


    #生成热力图，在后续工作中热力图会和转化后的图片一起送入沙漏网络中
    def create_heatmap_representations(self, source_image, kp_driving, kp_source):

        #提取图像高度和宽度信息
        spatial_size = source_image.shape[2:]

        #kp2gaussian将关键点转化为高斯形式表示
        gaussian_driving = kp2gaussian(kp_driving['fg_kp'], spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source['fg_kp'], spatial_size=spatial_size, kp_variance=self.kp_variance)
        #得到热力图
        heatmap = gaussian_driving - gaussian_source

        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type()).to(heatmap.device)
        heatmap = torch.cat([zeros, heatmap], dim=1)

        return heatmap

    def create_transformations(self, source_image, kp_driving, kp_source, bg_param):
        # K TPS transformaions
        """

        Args:
            source_image: 源图像
            kp_driving: 驱动关键点
            kp_source: 源关键点
            bg_param: 背景参数

        Returns:该函数的作用是得到源图像k个TPS变换的方式以及源图像背景仿射变换的方式并将其拼接起来
        也就是论文模型结构的 k+1 Transformation部分

        """
        bs, _, h, w = source_image.shape
        #获取驱动图关键点数据
        kp_1 = kp_driving['fg_kp']
        #获取源图像关键点数据
        kp_2 = kp_source['fg_kp']

        #将关键点数据的形状修改一下，这里的5是每个TPS要用到五个关键点，2是每个关键点由两个坐标表示
        kp_1 = kp_1.view(bs, -1, 5, 2)
        kp_2 = kp_2.view(bs, -1, 5, 2)

        #根据关键点数据做TPS变换，先想一想TPS变换是做什么的
        #给定原来的点和发生移动的点，需要根据这两个信息拟合一个新的面
        #这个面满足拟合距离与扭曲距离加权和最小
        #从数学推导出的结论来言，新平面是由 简单平面+弯曲 这样的方式生成的
        #当有了这个新平面之后，给定原来的点就可以知道移动完之后的点在哪了
        trans = TPS(mode = 'kp', bs = bs, kp_1 = kp_1, kp_2 = kp_2)

        #得到源图像TPS变换的相关参数
        driving_to_source = trans.transform_frame(source_image)

        #TODO:身份网络的作用需要再确认一下
        #生成身份网络，身份网络包含像素位置坐标，其作用是用于仿射变换
        identity_grid = make_coordinate_grid((h, w), type=kp_1.type()).to(kp_1.device)
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)

        # affine background transformation
        # 背景仿射变换
        if not (bg_param is None):

            #将身份网络转化为齐次坐标形式，所谓的非其次坐标指的是笛卡尔坐标(x,y)转化为齐次坐标后形式变为(x,y,1)其中1是常见的取值
            #这个操作是应用仿射变换的准备工作
            identity_grid = to_homogeneous(identity_grid)

            #应用仿射变换矩阵(背景参数)到身份网络上
            identity_grid = torch.matmul(bg_param.view(bs, 1, 1, 1, 3, 3), identity_grid.unsqueeze(-1)).squeeze(-1)
            #再把形式给转回去
            identity_grid = from_homogeneous(identity_grid)

        #将TPS变换得到的图像和仿射变换后的背景拼接起来，形成一个包含所有变换的张量
        transformations = torch.cat([identity_grid, driving_to_source], dim=1)
        return transformations

    #这段代码用来创建一个变形的源图像
    def create_deformed_source_image(self, source_image, transformations):

        bs, _, h, w = source_image.shape
        #对源图像进行升维、复制以及修改形状等操作
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_tps + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_tps + 1), -1, h, w)

        #对k+1个变换矩阵形状进行修改
        transformations = transformations.view((bs * (self.num_tps + 1), h, w, -1))

        # grid_sample用于根据给定的网格和输入数据执行采样操作。这个函数通常用于图像的仿射变换、透视变换等。
        # 在PyTorch中，grid_sample函数接受输入数据和一个对应的网格（grid），然后根据网格对输入数据进行采样，生成变换后的输出。
        deformed = F.grid_sample(source_repeat, transformations, align_corners=True)
        deformed = deformed.view((bs, self.num_tps+1, -1, h, w))
        return deformed

    def dropout_softmax(self, X, P):
        '''
        Dropout for TPS transformations. Eq(7) and Eq(8) in the paper.
        '''

        #根据dropout概率生成一个掩码，drop的大小可能是[batch_size,num_tps]
        drop = (torch.rand(X.shape[0],X.shape[1]) < (1-P)).type(X.type()).to(X.device)
        #表示总是会保留第一个变换的，从理解上来说是背景仿射变换
        drop[..., 0] = 1
        drop = drop.repeat(X.shape[2],X.shape[3],1,1).permute(2,3,0,1)

        # 下面的一系列操作都是根据掩码对TPS做softmax的过程
        maxx = X.max(1).values.unsqueeze_(1)
        X = X - maxx
        X_exp = X.exp()
        X[:,1:,...] /= (1-P)
        mask_bool =(drop == 0)
        X_exp = X_exp.masked_fill(mask_bool, 0)
        partition = X_exp.sum(dim=1, keepdim=True) + 1e-6
        return X_exp / partition

    def forward(self, source_image, kp_driving, kp_source, bg_param = None, dropout_flag=False, dropout_p = 0):

        #缩放因子不为1，对图像进行下采样
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        #这个字典用来存储前向传播的输出结果
        out_dict = dict()

        #创建关键点的热图表示，回顾一下关键点热图的生成过程：
        #首先是根据提取出的关键点将其转化为高斯形式的表示，驱动图像表示-源图像表示并与零矩阵连接形成热力图
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)

        #得到k个TPS变换和1个背景仿射变换的结果
        transformations = self.create_transformations(source_image, kp_driving, kp_source, bg_param)

        #根据得到的结果对源图片做变换
        deformed_source = self.create_deformed_source_image(source_image, transformations)

        #将变换后的源图像写到字典中
        out_dict['deformed_source'] = deformed_source
        # out_dict['transformations'] = transformations

        deformed_source = deformed_source.view(bs,-1,h,w)

        #将热图表示和变形后的源图像拼接
        input = torch.cat([heatmap_representation, deformed_source], dim=1)
        input = input.view(bs, -1, h, w)

        #将输入放进沙漏网络中
        prediction = self.hourglass(input, mode = 1)

        #用卷积神经网络处理沙漏网络的最后一个输出结果，得到贡献图
        contribution_maps = self.maps(prediction[-1])

        #如果需要有dropout，就用带掩码的softmax处理贡献图
        #没有的话直接用softmax处理
        if(dropout_flag):
            contribution_maps = self.dropout_softmax(contribution_maps, dropout_p)
        else:
            contribution_maps = F.softmax(contribution_maps, dim=1)
        out_dict['contribution_maps'] = contribution_maps

        # Combine the K+1 transformations
        # Eq(6) in the paper
        contribution_maps = contribution_maps.unsqueeze(2)
        transformations = transformations.permute(0, 1, 4, 2, 3)

        #计算每个变换贡献图的加权和，得到最终的光流(变换)
        deformation = (transformations * contribution_maps).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        #将光流添加到字典中
        out_dict['deformation'] = deformation # Optical Flow

        #创建空列表用于存储遮挡
        occlusion_map = []

        if self.multi_mask:

            #遍历遮挡模块
            #这里先这样整理理解，occlusion[i]其实是一个卷积层
            #对于prediction(x)的返回值是解码器输出的列表，相当于取一部分的解码器输出做卷积
            for i in range(self.occlusion_num-self.up_nums):
                occlusion_map.append(torch.sigmoid(self.occlusion[i](prediction[self.up_nums-self.occlusion_num+i])))

            #将解码器的最后一个输出作为预测输出
            prediction = prediction[-1]

            #对于预测输出做上采样，在上采样之后将其结果加入到occlusion_map中
            for i in range(self.up_nums):
                prediction = self.up[i](prediction)
                occlusion_map.append(torch.sigmoid(self.occlusion[i+self.occlusion_num-self.up_nums](prediction)))

        #如果不需要使用多重掩码的话，直接就是将网络最后一个输出拿出来，做个卷积激活一下输出就可以了
        else:
            occlusion_map.append(torch.sigmoid(self.occlusion[0](prediction[-1])))
                
        out_dict['occlusion_map'] = occlusion_map # Multi-resolution Occlusion Masks

        #forward函数的返回结果是我们创建的字典，包括光流以及Multi-resolution Occlusion Masks(多分辨率遮挡掩码)
        return out_dict
