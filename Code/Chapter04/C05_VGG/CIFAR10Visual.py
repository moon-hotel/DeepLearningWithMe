"""
文件名: Code/Chapter04/C05_VGG/CIFAR10Visual.py
创建时间: 2023/3/27 20:44 下午
"""

from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = CIFAR10(root='~/Datasets/CIFAR10', train=False,
                      download=True)
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    print(dataset.data[0].shape)
    print(dataset.targets[0])
    row = 2
    col = 3
    plt.subplots(2, 3)
    for i in range(row * col):
        plt.subplot(row, col, i + 1)
        plt.xlabel(labels[dataset.targets[i]],fontsize=13)
        plt.imshow(dataset.data[i])
    plt.tight_layout()
    plt.show()
