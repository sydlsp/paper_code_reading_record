import os
import json

w=640
h=512

label_path="result/labels_ir_best"

json_file_path="result/json_result/labels_ir_best.json"

json_file=open(json_file_path,'w',encoding='utf-8')

final_list=[]

for i in range(1,1001):
    file_name='{:0>5d}'.format(i)+".txt"

    print(file_name)
    if file_name in os.listdir(label_path):

        print("here")
        file = open(os.path.join(label_path, file_name))

        # 开始读文件中的每一行
        for line in file:
            # line一开始是字符串形式的，先分割转化为字符串列表，但现在里面的数字还是字符串形式
            line = line.split()
            # 把字符串转化为数字
            line = [float(line_child) for line_child in line]

            bbox_x = float(format((line[1] - (line[3] / 2)) * w, '.4f'))
            bbox_y = float(format((line[2] - (line[4] / 2)) * h, '.4f'))
            bbox_h = float(format(line[3] * w, '.4f'))
            bbox_w = float(format(line[4] * h, '.4f'))

            dict = {"image_id": int(file_name[:5]), "category_id": int(line[0] + 1),
                    "bbox": [bbox_x, bbox_y, bbox_h, bbox_w], "score": float(format(line[5], '.2f'))}

            final_list.append(dict)
    else:
        print("there")
        dict={"image_id": int(file_name[:5]), "category_id": None,
                    "bbox": None, "score": None}
        final_list.append(dict)

json.dump(final_list,json_file)

print("ok")

# # 遍历每一个文件
# for file_name in os.listdir(label_path):
#
#     # 读文件内容
#     file=open(os.path.join(label_path,file_name))
#
#     # 开始读文件中的每一行
#     for line in file:
#
#         # line一开始是字符串形式的，先分割转化为字符串列表，但现在里面的数字还是字符串形式
#         line=line.split()
#         # 把字符串转化为数字
#         line=[float(line_child) for line_child in line]
#
#         bbox_x=float(format((line[1]-(line[3]/2))*w,'.4f'))
#         bbox_y=float(format((line[2]-(line[4]/2))*h,'.4f'))
#         bbox_w=float(format(line[3]*w,'.4f'))
#         bbox_h=float(format(line[4]*h,'.4f'))
#
#         dict={"image_id":file_name[:5],"category_id":int(line[0]+1),
#               "bbox":[bbox_x,bbox_y,bbox_h,bbox_w],"score":float(format(line[5],'.2f'))}
#
#         final_list.append(dict)
#
# json.dump(final_list,json_file)
#
# print("ok")







