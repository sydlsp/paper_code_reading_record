import os.path
import sys

"""
还是同样的操作
"""
__dir__=os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0,os.path.abspath(os.path.join(__dir__, "..")))

for i in sys.path:
    print(i)


from level_A.b_1 import B_1 #同理

b=B_1()
b.call_A_1()