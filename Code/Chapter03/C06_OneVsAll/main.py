"""
文件名: Code/Chapter03/C01_OP/main.py
创建时间: 2023/1/18 21:52 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


def visualization():
    num_points = 200
    centers = [[1, 1], [2, 2], [1, 3]]  # 指定中心
    x, y = make_blobs(n_samples=num_points, centers=centers,
                      cluster_std=0.15, random_state=np.random.seed(10))
    index_c0, index_c1, index_c2 = (y == 0), (y == 1), (y == 2)
    c0, c1, c2 = x[index_c0], x[index_c1], x[index_c2]
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.scatter(c0[:, 0], c0[:, 1], marker='s',color='blue', label='类别1', s=50)
    plt.scatter(c1[:, 0], c1[:, 1], marker='o',color='orange', label='类别2', s=50)
    plt.scatter(c2[:, 0], c2[:, 1], marker='v',color='green', label='类别3', s=50)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()


def visualization_ova():
    plt.figure(figsize=(12, 3.5), dpi=80)
    plt.subplot(1, 3, 1)
    num_points = 200
    centers = [[1, 1], [2, 2], [1, 3]]  # 指定中心
    x, y = make_blobs(n_samples=num_points, centers=centers,
                      cluster_std=0.15, random_state=np.random.seed(10))
    index_c0, index_c1, index_c2 = (y == 0), (y == 1), (y == 2)
    c0, c1, c2 = x[index_c0], x[index_c1], x[index_c2]
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.scatter(c0[:, 0], c0[:, 1], marker='s', color='blue', label='类别1', s=50)
    plt.scatter(c1[:, 0], c1[:, 1], marker='*', color='gray', label='类别0', s=50)
    plt.scatter(c2[:, 0], c2[:, 1], marker='*', color='gray', label='类别0', s=50)
    plt.plot([0.65, 2.0], [2.6, 0.85])
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.legend(fontsize=14)

    plt.subplot(1, 3, 2)
    num_points = 200
    centers = [[1, 1], [2, 2], [1, 3]]  # 指定中心
    x, y = make_blobs(n_samples=num_points, centers=centers,
                      cluster_std=0.15, random_state=np.random.seed(10))
    index_c0, index_c1, index_c2 = (y == 0), (y == 1), (y == 2)
    c0, c1, c2 = x[index_c0], x[index_c1], x[index_c2]
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.scatter(c0[:, 0], c0[:, 1], marker='*', color='gray', label='类别0', s=50)
    plt.scatter(c1[:, 0], c1[:, 1], marker='o', color='orange', label='类别1', s=50)
    plt.scatter(c2[:, 0], c2[:, 1], marker='*', color='gray', label='类别0', s=50)
    plt.plot([1.45, 1.55], [0.2, 3.5])
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.legend(fontsize=14)

    plt.subplot(1, 3, 3)
    num_points = 200
    centers = [[1, 1], [2, 2], [1, 3]]  # 指定中心
    x, y = make_blobs(n_samples=num_points, centers=centers,
                      cluster_std=0.15, random_state=np.random.seed(10))
    index_c0, index_c1, index_c2 = (y == 0), (y == 1), (y == 2)
    c0, c1, c2 = x[index_c0], x[index_c1], x[index_c2]
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.scatter(c0[:, 0], c0[:, 1], marker='*', color='gray', label='类别0', s=50)
    plt.scatter(c1[:, 0], c1[:, 1], marker='*', color='gray', label='类别0', s=50)
    plt.scatter(c2[:, 0], c2[:, 1], marker='v', color='green', label='类别1', s=50)
    plt.plot([0.66, 2.2], [1, 3.1])
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.legend(fontsize=14)
    plt.tight_layout()  # 调整子图间距
    plt.show()


if __name__ == '__main__':
    visualization()
    visualization_ova()
