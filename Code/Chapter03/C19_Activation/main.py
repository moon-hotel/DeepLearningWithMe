"""
文件名: Code/Chapter03/C19_Activation/main.py
创建时间: 2023/2/18 10:42 上午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


class MySigmoid(nn.Module):
    def forward(self, x):
        return sigmoid(x)


def plot_sigmoid():
    x = np.linspace(-8, 8, 100)
    y = 1 / (1 + np.exp(-x))
    y_prime = y * (1 - y)

    a = plt.gca()
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['left'].set_position('center')
    # a.spines['bottom'].set_position('center')
    plt.hlines(1, -8, 8, linestyle='--', color='black')
    plt.hlines(0, -8, 8, linestyle='--', color='black')
    plt.plot(x, y, label='$g(x)$', linestyle='-', color='black')
    plt.plot(x, y_prime, label='$g^{\prime}(x)$', linestyle='--', color='black')
    plt.legend(loc='upper left', fontsize=13)
    plt.show()


def test_Sigmoid():
    x = torch.randn([2, 5], dtype=torch.float32)
    net = nn.Sequential(MySigmoid())
    y = net(x)
    print(f"Sigmoid前: {x}")
    print(f"Sigmoid后: {y}")


def plot_tanh():
    x = np.linspace(-8, 8, 100)
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    y_prime = 1 - y ** 2

    a = plt.gca()
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['left'].set_position('center')
    a.spines['bottom'].set_position('center')

    plt.plot(x, y, label='$g(x)$', linestyle='-', color='black')
    plt.plot(x, y_prime, label='$g^{\prime}(x)$', linestyle='--', color='black')
    plt.legend(loc='upper left', fontsize=13)
    plt.show()


def tanh(x):
    p = torch.exp(x) - torch.exp(-x)
    q = torch.exp(x) + torch.exp(-x)
    return p / q


class MyTanh(nn.Module):
    def forward(self, x):
        return tanh(x)


def test_Tanh():
    x = torch.randn([2, 5], dtype=torch.float32)
    net = nn.Sequential(MyTanh())
    y = net(x)
    print(f"Tanh前: {x}")
    print(f"Tanh后: {y}")


def plot_relu():
    x = np.linspace(-3, 3, 200)
    mask = x >= 0.
    y = x * mask
    y_prime = (x > 0) * 1

    a = plt.gca()
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    # a.spines['left'].set_position('center')
    a.spines['left'].set_position(('data', 0))
    a.spines['bottom'].set_position(('data', 0.5))

    plt.plot(x, y, label='$g(x)$', linestyle='-', color='black')
    plt.plot(x, y_prime, label='$g^{\prime}(x)$', linestyle='--', color='black')
    plt.legend(loc='upper left', fontsize=13)
    plt.show()


def relu(x):
    mask = x >= 0.
    return x * mask


class MyReLU(nn.Module):
    def forward(self, x):
        return relu(x)


def test_ReLU():
    x = torch.randn([2, 5], dtype=torch.float32)
    net = nn.Sequential(MyReLU())
    y = net(x)
    print(f"ReLu前: {x}")
    print(f"ReLU后: {y}")


def plot_leakyrelu():
    x = np.linspace(-3, 3, 200)
    gamma = 0.2
    y = (x >= 0) * x + gamma * (x < 0) * x
    y_prime = (x > 0) + gamma * (x <= 0)
    a = plt.gca()
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    # a.spines['left'].set_position('center')
    a.spines['left'].set_position(('data', 0))
    a.spines['bottom'].set_position(('data', 0.5))

    plt.plot(x, y, label='$g(x)$', linestyle='-', color='black')
    plt.plot(x, y_prime, label='$g^{\prime}(x)$', linestyle='--', color='black')
    plt.legend(loc='upper left', fontsize=13)
    plt.show()


def leakyrelu(x, gamma=0.2):
    y = (x >= 0) * x + gamma * (x < 0) * x
    return y


class MyLeakyReLU(nn.Module):
    def __init__(self, gamma=0.2):
        super(MyLeakyReLU, self).__init__()
        self.gamma = gamma

    def forward(self, x):
        return leakyrelu(x, self.gamma)


def test_LeakyReLU():
    x = torch.randn([2, 5], dtype=torch.float32)
    net = nn.Sequential(MyLeakyReLU(0.2))
    y = net(x)
    print(f"LeakyReLU前: {x}")
    print(f"LeakyReLU后: {y}")


if __name__ == '__main__':
    # plot_sigmoid()
    # test_Sigmoid()

    # plot_tanh()
    # test_Tanh()

    # plot_relu()
    # test_ReLU()

    # plot_leakyrelu()
    test_LeakyReLU()
