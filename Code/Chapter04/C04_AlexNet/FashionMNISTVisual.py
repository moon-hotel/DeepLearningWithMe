"""
文件名: Code/Chapter04/C04_AlexNet/FashionMNISTVisual.py
创建时间: 2023/3/25 7:55 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    data = FashionMNIST(root='~/Datasets/FashionMNIST', train=False,
                        download=True, transform=transforms.ToTensor())
    index = 2
    print(f"输入向量为：{data[index][0].reshape(-1)}")
    print(f"标签为：{classes[data[index][1]]}")
    plt.imshow(data[index][0].reshape([28, 28]), cmap=plt.cm.gray_r, interpolation="nearest")
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.show()
