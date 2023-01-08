"""
文件名: Code/Chapter03/C01_OP/main.py
创建时间: 2023/1/7 10:52 上午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn


def test_linear():
    x = torch.tensor([[1., 2, 3, 4], [4, 5, 6, 7]], dtype=torch.float32)  # [2,4]
    layer = nn.Linear(4, 5)  #
    y = layer(x)
    print(y)


def multi_layers():
    x = torch.tensor([[1., 2, 3, 4], [4, 5, 6, 7]], dtype=torch.float32)  # [2,4]
    layer1 = nn.Linear(4, 5)  #
    layer2 = nn.Linear(5, 3)
    layer3 = nn.Linear(3, 1)
    y1 = layer1(x)
    y2 = layer2(y1)
    y3 = layer3(y2)
    print(y3)


def multi_layers_sequential():
    x = torch.tensor([[1., 2, 3, 4], [4, 5, 6, 7]], dtype=torch.float32)  # [2,4]
    net = nn.Sequential(nn.Linear(4, 5),
                        nn.Linear(5, 3),
                        nn.Linear(3, 1))
    y = net(x)
    print(y)


def test_loss():
    """
    J(w,b)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{({{y}^{(i)}}-{{{\hat{y}}}^{(i)}})}^{2}}}
    :return:
    """
    y = torch.tensor([1, 2, 3], dtype=torch.float32)
    y_hat = torch.tensor([2, 2, 1], dtype=torch.float32)
    l1 = 0.5 * torch.mean((y - y_hat) ** 2)
    print(l1)
    print(l1 * 2)
    loss = nn.MSELoss(reduction='mean')
    l2 = loss(y, y_hat)
    print(l2)


if __name__ == '__main__':
    test_linear()
    multi_layers()
    multi_layers_sequential()
    test_loss()
