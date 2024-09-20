from .src.a_1 import A_1

# 当我们想导入”包“内的相关文件，用相对地址的写法

class B_1:
    def __init__(self):
        pass
    def call_A_1(self):
        a = A_1()
        a.print_class()