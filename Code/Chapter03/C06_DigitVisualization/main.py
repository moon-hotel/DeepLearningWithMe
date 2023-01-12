"""
文件名: Code/Chapter03/C06_DigitVisualization/main.py
创建时间: 2023/1/12 8:42 下午
"""

from sklearn.datasets import load_digits

import matplotlib.pyplot as plt

if __name__ == '__main__':
    digits = load_digits()
    print(f"标签为：{digits.target[-1]}")
    print(f"输入向量为：{digits.images[-1].reshape(-1)}")
    plt.imshow(digits.images[-1], cmap=plt.cm.gray_r,interpolation="nearest")
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.show()
