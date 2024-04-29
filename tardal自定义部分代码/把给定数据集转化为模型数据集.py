import os
import shutil
from PIL import Image

"""
mydata是给定的数据集文件夹
flod是要转换到的新数据集文件夹
"""

fold_path="./data_all"
fold_ir_path=os.path.join(fold_path,"ir")
fold_vi_path=os.path.join(fold_path,"vi")
fold_meta_path=os.path.join(fold_path,"meta")

if not os.path.exists(fold_path):
    os.makedirs(fold_path)

if not os.path.exists(fold_ir_path):
    os.makedirs(fold_ir_path)

if not os.path.exists(fold_vi_path):
    os.makedirs(fold_vi_path)

if not os.path.exists(fold_meta_path):
    os.makedirs(fold_meta_path)

"""
确定文件夹路径
"""
mydata_train_rgb_path="./mydata/train/rgb"
mydata_train_tir_path="./mydata/train/tir"

mydata_val_rgb_path="./mydata/val/rgb"
mydata_val_tir_path="./mydata/val/tir"

mydata_test_rgb_path="./mydata/test/rgb"
mydata_test_tir_path="./mydata/test/tir"


"""
从文件夹中读取文件列表
"""
mydata_train_rgb_filelist=os.listdir(mydata_train_rgb_path)
mydata_train_tir_filelist=os.listdir(mydata_train_tir_path)

mydata_val_rgb_filelist=os.listdir(mydata_val_rgb_path)
mydata_val_tir_filelist=os.listdir(mydata_val_tir_path)

mydata_test_rgb_filelist=os.listdir(mydata_test_rgb_path)
mydata_test_tir_filelist=os.listdir(mydata_test_tir_path)

"""
从下面开始用到什么代码就把注释给去掉，不用就注释掉
"""

"""
把train文件夹中的数据转移到新文件夹中
"""
# for i in range(0,len(mydata_train_rgb_filelist)):
#     if (i%500==0):
#         print(i)
#     image_rgb=Image.open(os.path.join(mydata_train_rgb_path,mydata_train_rgb_filelist[i]))
#     image_vir=Image.open(os.path.join(mydata_train_tir_path,mydata_train_tir_filelist[i]))
#     add_path='{:0>5d}'.format(i+1)+".jpg"
#     image_rgb.save(os.path.join(fold_vi_path,add_path))
#     image_vir.save(os.path.join(fold_ir_path,add_path))


"""把val文件夹中的数据放到train数据的后面"""

count=len(mydata_train_rgb_filelist)

# for i in range(0,len(mydata_val_rgb_filelist)):
#     if (i % 500 == 0):
#         print(i)
#     image_rgb=Image.open(os.path.join(mydata_val_rgb_path,mydata_val_rgb_filelist[i]))
#     image_vir=Image.open(os.path.join(mydata_val_tir_path,mydata_val_tir_filelist[i]))
#     add_path='{:0>5d}'.format(i+count+1)+".jpg"
#     image_rgb.save(os.path.join(fold_vi_path,add_path))
#     image_vir.save(os.path.join(fold_ir_path,add_path))

"""
把test文件夹中的数据放到val后面
"""
count_1=len(mydata_train_rgb_filelist)+len(mydata_val_rgb_filelist)

# for i in range(0,len(mydata_test_rgb_filelist)):
#     if (i % 500 == 0):
#         print(i)
#     image_rgb=Image.open(os.path.join(mydata_test_rgb_path,mydata_test_rgb_filelist[i]))
#     image_vir=Image.open(os.path.join(mydata_test_tir_path,mydata_test_tir_filelist[i]))
#     add_path='{:0>5d}'.format(i+count_1+1)+".jpg"
#     image_rgb.save(os.path.join(fold_vi_path,add_path))
#     image_vir.save(os.path.join(fold_ir_path,add_path))

"""
构造meta中的txt文件
"""

# meta_pred_file=open(os.path.join(fold_meta_path,"pred.txt"),'w',encoding='utf-8')
# meta_train_file=open(os.path.join(fold_meta_path,"train.txt"),'w',encoding='utf-8')
# meta_val_file=open(os.path.join(fold_meta_path,"val.txt"),'w',encoding='utf-8')
# meta_test_file=open(os.path.join(fold_meta_path,"test.txt"),'w',encoding='utf-8')
#
# for i in range(0,count_1+len(mydata_test_rgb_filelist)):
#     meta_pred_file.write('{:0>5d}'.format(i+1)+".jpg\n")
#
# for i in range(0,len(mydata_train_rgb_filelist)):
#     meta_train_file.write('{:0>5d}'.format(i+1)+".jpg\n")
#
# for i in range(0,len(mydata_val_rgb_filelist)):
#     meta_val_file.write('{:0>5d}'.format(i+count+1)+".jpg\n")
#
# for i in range(0,len(mydata_test_rgb_filelist)):
#     meta_test_file.write('{:0>5d}'.format(i+count_1+1)+".jpg\n")

"""
将coco数据集格式转化为yolov5格式见 将coco格式数据集转化为yolov5格式.py
"""

"""
下面做的事情是把yolo格式的txt数据转移到label文件夹中
"""

fold_label_path=os.path.join(fold_path,"labels")

if not os.path.exists(fold_label_path):
    os.makedirs(fold_label_path)

# 把yolo/train 中的文件复制到label中

# for file_name in os.listdir(os.path.join(fold_path,"yolo","train")):
#     if file_name!="classes.txt" and file_name!="train2017.txt":
#         shutil.copy(os.path.join(fold_path,"yolo","train",file_name),os.path.join(fold_label_path,file_name))

# 把yolo/val 中的文件复制到label中，接到train文件的后面
# i=len(mydata_train_rgb_filelist)
#
# for file_name in os.listdir(os.path.join(fold_path,"yolo","val")):
#     if file_name != "classes.txt" and file_name != "train2017.txt":
#          shutil.copy(os.path.join(fold_path,"yolo","val",file_name),os.path.join(fold_label_path,'{:0>5d}'.format(i+1)+".txt"))
#
#     i=i+1

# 这里似乎labels文件还需要test文件的label但是我没有啊
for i in range(0,len(mydata_test_rgb_filelist)):
    f=open(os.path.join(fold_path,"labels",'{:0>5d}'.format(i+len(mydata_train_rgb_filelist)+len(mydata_val_rgb_filelist)+1)+".txt"),'w',encoding='utf-8')
