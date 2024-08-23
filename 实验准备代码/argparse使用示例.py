import argparse

def parse_args():

    # 创建新的ArgumentParser对象(参数对象)
    parse=argparse.ArgumentParser(description="This is a test")
    # 往参数对象中添加参数
    parse.add_argument("--radius",type=int,help="The radius of the circle",default=10)
    parse.add_argument("--height",type=int,help="The height of the cylinder",default=20)

    # 解析参数
    args=parse.parse_args()
    return args

def cal_vol(radius:int,height:int):

    return 3.14*radius*radius*height

args=parse_args()
# 使用解析对象
print(cal_vol(args.radius,args.height))
