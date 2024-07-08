import numpy as np
import matplotlib.pyplot as plt


def f_non_convex(W):
    return 0.2 * W ** 2 + 2 * (np.sin(2 * W))


def f_non_convex_grad(W, ascent=False, learning_rate=0.2):
    grad = 0.4 * W + 2 * np.cos(2 * W) * 2
    if ascent:
        grad *= -1
    W = W - learning_rate * grad
    return grad, np.array([W, f_non_convex(W)])


def plot_countour():
    W = np.linspace(-5, 4, 800)
    J = f_non_convex(W)
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内

    plt.plot(W, J, c='black')
    # plt.scatter(-0.7584, -1.8747, marker='*', c='black', s=80, label='global optimum ')  # 非实际计算
    plt.scatter(-0.7584, -1.8747, marker='*', c='black', s=80, label='全局最优解')  # 非实际计算
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    p = np.array([-4.8, f_non_convex(-4.8)])  # 起始位置 1
    plt.annotate('A',(-4.5,4.8),fontsize=12)
    plt.scatter(p[0], p[1], c='black')
    plt.legend(fontsize=15, loc='upper center')
    for i in range(11):
        g, q = f_non_convex_grad(p[0], learning_rate=0.02)
        print("P{}:{},grad = {}".format(i, p, g))
        plt.arrow(p[0], p[1], q[0] - p[0], q[1] - p[1], head_width=0.1,
                  head_length=0.1, linewidth=2, color='black')
        p = q
        plt.scatter(p[0], p[1], c='black')


    p = np.array([-2., f_non_convex(-2.)])  # 起始位置 2
    plt.annotate('B',(-1.5,2.2),fontsize=12)
    plt.scatter(p[0], p[1], c='black',marker='s')
    plt.legend(fontsize=15, loc='upper center')
    for i in range(15):
        g, q = f_non_convex_grad(p[0], learning_rate=0.02)
        print("P{}:{},grad = {}".format(i, p, g))
        plt.arrow(p[0], p[1], q[0] - p[0], q[1] - p[1], head_width=0.1,
                  head_length=0.1, linewidth=2, color='black')
        p = q
        plt.scatter(p[0], p[1], c='black',marker='s')


    plt.ylabel('J(w)', fontsize=15)
    plt.xlabel('w', fontsize=15)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_countour()
