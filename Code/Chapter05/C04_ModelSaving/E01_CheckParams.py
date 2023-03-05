"""
文件名: Code/Chapter05/C04_ModelSaving/E01_CheckParams.py
创建时间: 2023/3/5 8:45 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import sys

sys.path.append("../../")
from Chapter04.C03_LeNet5.LeNet5 import LeNet5

if __name__ == '__main__':
    model = LeNet5()
    print("Model's state_dict:")
    for (name, param) in model.state_dict().items():
        print(name, param.size())
