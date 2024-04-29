import os
import shutil

ori_val_path= "yolov5data/val_ori"

val_path= "yolov5data/val"

if not os.path.exists(val_path):
    os.makedirs(val_path)

i=1

for file_name in os.listdir(ori_val_path):
    shutil.copy(os.path.join(ori_val_path,file_name),os.path.join(val_path,'{:0>5d}'.format(i)+".jpg"))
    i=i+1
