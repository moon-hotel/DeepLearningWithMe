import torch.nn as nn
import torch
import sys

sys.path.append('../../')
from Chapter04.C03_LeNet5.LeNet5 import LeNet5


def init():
    value = torch.ones((3, 5))
    print(value)
    nn.init.kaiming_normal_(value)
    print(value)


def init_lenet5():
    model = LeNet5()
    for p in model.parameters(): # 方法 1
        if p.dim() > 1:  # 偏置
            nn.init.xavier_uniform_(p)
    for m in model.modules():     # 方法 2
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)


if __name__ == '__main__':
    init()
    init_lenet5()
