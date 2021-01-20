import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot():
    x = np.linspace(-6, 6, 100)
    y = sigmoid(x)
    plt.plot(x, y, c='b')

    x = np.linspace(-2.5, 2.5, 10)
    y = 0.235 * x + 0.5
    plt.plot(x, y, c='b', linestyle='--')

    plt.vlines(-1.1, -0.1, 1.1, colors='r',linestyle='--')
    plt.vlines(1.1, -0.1, 1.1, colors='r',linestyle='--')
    plt.show()


if __name__ == '__main__':
    plot()
