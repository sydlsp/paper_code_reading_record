import os
import sys

"""
将当前文件夹以及上一级文件夹加到sys.path中，这个加入的过程是动态加的
"""
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

for i in sys.path:
    print(i)

from level_A.b_1 import B_1  # 导入不同包内的文件用这种相对地址的写法
b=B_1()
b.call_A_1()