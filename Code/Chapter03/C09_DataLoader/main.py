"""
文件名: Code/Chapter03/C09_DataLoader/main.py
创建时间: 2023/2/1 20:39 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset
import torch
import numpy as np


def DataLoader1():
    data_loader = MNIST(root='~/Datasets/MNIST',
                               download=True,
                               transform=torchvision.transforms.ToTensor())
    data_iter = DataLoader(data_loader, batch_size=32)
    for (x, y) in data_iter:
        print(x.shape, y.shape)
        break


def DataLoader2():
    x = torch.tensor(np.random.random([100, 3, 16, 16]))
    y = torch.tensor(np.random.randint(0, 10, 100))
    dataset = TensorDataset(x, y)
    data_iter = DataLoader(dataset, batch_size=32)
    for (x, y) in data_iter:
        print(x.shape, y.shape)
        break


if __name__ == '__main__':
    # DataLoader1()
    DataLoader2()
