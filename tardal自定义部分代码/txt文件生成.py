import os
import random

file_num=600

train_total=2/3

dir_path="./M3FD/M3FD_Detection/meta"

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

pred_txt=os.path.join(dir_path,"pred.txt")

train_txt=os.path.join(dir_path,"train.txt")

val_txt=os.path.join(dir_path,"val.txt")

pred_f=open(pred_txt,'w',encoding='utf_8')

train_f=open(train_txt,'w',encoding='utf-8')

val_f=open(val_txt,'w',encoding='utf-8')

for i in range(0,file_num):

    i_str='{:0>5d}'.format(i)
    str_w=i_str+".png"+"\n"
    pred_f.write(str_w)

    if (random.random()<train_total):
        train_f.write(str_w)
    else:
        val_f.write(str_w)

print("ok")



