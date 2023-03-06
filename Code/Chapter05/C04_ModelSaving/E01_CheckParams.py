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

    # Model's state_dict:
    # conv.0.weight torch.Size([6, 1, 5, 5])
    # conv.0.bias torch.Size([6])
    # conv.3.weight torch.Size([16, 6, 5, 5])
    # conv.3.bias torch.Size([16])
    # fc.1.weight torch.Size([120, 400])
    # fc.1.bias torch.Size([120])
    # fc.3.weight torch.Size([84, 120])
    # fc.3.bias torch.Size([84])
    # fc.5.weight torch.Size([10, 84])
    # fc.5.bias torch.Size([10])
