"""
文件名: Code/Chapter03/C06_DigitVisualization/main.py
创建时间: 2023/1/12 8:42 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms



def load_dataset():
    data = MNIST(root='~/Datasets/MNIST', train=True, download=True,
                 transform=transforms.ToTensor())
    return data


import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = load_dataset()
    f_img = np.round(data[0][0].reshape(-1).numpy(),3)
    print(f"标签为：{data[0][1]}")
    print(f"输入向量为：{f_img}")
    plt.imshow(data[0][0].reshape([28, 28]), cmap=plt.cm.gray_r, interpolation="nearest")
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.show()
