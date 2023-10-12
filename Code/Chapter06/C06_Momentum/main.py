"""
文件名: Code/Chapter03/C08_SGDVisualization/main.py
创建时间: 2023/2/1 20:10 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import numpy as np
import matplotlib.pyplot as plt


def f_grad(W1, W2, descent=True, momentum=0., last_grad=0.):
    grad = np.array([1 / 3 * W1, 4 * W2])
    if descent:
        last_grad *= -1  # 梯度下架时上次返回的是梯度的反方向，所以这种情况下需要先还原
    grad = momentum * last_grad + grad
    if descent:
        grad *= -1
    return grad


def plot_countour():
    W1 = np.arange(-10, 10, 0.25)
    W2 = np.arange(-50, 50, 0.25)

    W1, W2 = np.meshgrid(W1, W2)
    J = (1 / 6) * W1 ** 2 + 2 * W2 ** 2

    plt.figure(figsize=(12, 6), dpi=80)
    plt.subplot(1, 2, 1)
    plt.ylim(-50, 50)
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    # CS = plt.contour(W , W2, J, 10, colors='black')
    CS = plt.contour(W1, W2, J, 10)
    plt.clabel(CS, inline=2, fontsize=10)
    plt.scatter(0, 0, s=60, c='black')
    plt.xlabel(r'$w_1$', fontsize=15)
    plt.ylabel(r'$w_2$', fontsize=15)
    learning_rate = 0.45
    # plt.title(f"Learning rate = {learning_rate}, Iter = 20", fontsize=15)
    plt.title(f"Learning rate = {learning_rate}, Iter = 10, Momentum = 0.4", fontsize=15)
    p = np.array([-10, -25.])  # 起始位置
    plt.scatter(p[0], p[1], c='black')
    grad = 0.
    for i in range(10):  # 梯度反方向，最速下降曲线
        # q = learning_rate * f_grad(p[0], p[1])
        grad = f_grad(p[0], p[1], momentum=0.4, last_grad=grad)
        q = learning_rate * grad
        # print("P{}:{}".format(i, p))
        # plt.arrow(p[0], p[1], q[0], q[1], head_width=0.1, head_length=0.05, fc='black', ec='black')
        plt.arrow(p[0], p[1], q[0], q[1], head_width=0.25, head_length=0.05)
        p += q  # 上一次的位置加上本次的梯度

    plt.subplot(1, 2, 2)
    plt.ylim(-50, 50)
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    # CS = plt.contour(W , W2, J, 10, colors='black')
    CS = plt.contour(W1, W2, J, 10)
    plt.clabel(CS, inline=2, fontsize=10)
    plt.scatter(0, 0, s=60, c='black')
    plt.xlabel(r'$w_1$', fontsize=15)
    plt.ylabel(r'$w_2$', fontsize=15)
    plt.title("Mini-batch Gradient Descent", fontsize=15)
    p = np.array([-10, -25.])  # 起始位置
    learning_rate = 0.55
    plt.title(f"Learning rate = {learning_rate}, Iter = 5, Momentum = 0.55", fontsize=15)
    plt.scatter(p[0], p[1], c='black')
    grad = 0.
    for i in range(5):  # 梯度反方向，最速下降曲线
        grad = f_grad(p[0], p[1], momentum=0.55, last_grad=grad)
        q = learning_rate * grad
        plt.arrow(p[0], p[1], q[0], q[1], head_width=0.25, head_length=0.05)
        # plt.arrow(p[0], p[1], q[0], q[1], head_width=0.25, head_length=0.05, ec='red', fc='red', linestyle='--')
        p += q  # 上一次的位置加上本次的梯度
    plt.tight_layout()  # 调整子图间距
    plt.show()


if __name__ == '__main__':
    plot_countour()
