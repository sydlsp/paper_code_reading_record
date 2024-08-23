import pytorch3d
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    TexturesVertex
)
from pytorch3d.renderer import look_at_view_transform
import matplotlib.pyplot as plt
import os

obj_file_path="./obj_file/test.obj"
print(os.path.exists(obj_file_path))

# 加载obj文件
obj_file=load_obj(obj_file_path)
# verts包含了模型的顶点坐标，faces_idx包含了模型的面信息，通常是顶点索引的列表，aux包含辅助信息(如纹理坐标和法线等)
verts, faces_idx,aux=obj_file
faces=faces_idx.verts_idx

# 我们的obj文件现在其实是没有辅助信息的
# 那我们下面要补充一下这种信息
verts_rgb=torch.ones_like(verts)[None]
textures=TexturesVertex(verts_features=verts_rgb.to("cuda"))

# 创建Mesh对象
mesh=Meshes(verts=[verts], faces=[faces],textures=textures)
"""
检查一下mesh对象是否有纹理
"""
# has_textures=mesh.textures is not None and mesh.textures.maps_packed().shape[0]>0
#
# print(f"Mesh has textures: {has_textures}")

# 设置相机参数
# 旋转矩阵R,平移向量T  dist：相机到物体的距离，elev：相机的仰角，azim：相机的方位角
R,T=look_at_view_transform(dist=2.7,elev=10,azim=330)

# 根据相机参数，创建一个透视相机对象
cameras=FoVPerspectiveCameras(device="cuda",R=R,T=T)

# 设置光照
lights=PointLights(device="cuda",location=[[0.0,0.0,3.0]])

# 设置渲染器
raster_settings=RasterizationSettings(
    image_size=256, # 渲染出图像的大小
    blur_radius=0.0, # 光栅化的时候不应用模糊
    faces_per_pixel=1, # 每个像素最多显示一个面
)

render=MeshRenderer(
    # 光栅化器，用于将3D网格对象转换为2D图像
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings,
    ),
    # 着色器，负责计算光照和颜色
    shader=HardPhongShader(
        device="cuda",
        cameras=cameras,
        lights=lights,
    )
)

# 渲染Mesh成2D图像
images=render(mesh.to("cuda"),)

plt.figure(figsize=(10,10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.show()





