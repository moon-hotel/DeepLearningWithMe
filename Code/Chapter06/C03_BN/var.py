"""
文件名: Code/Chapter06/C03_BN/var.py
创建时间: 2023/4/4 3:26 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
activations = list(np.random.randn(5000) * 2)

#
print("真实期望:", np.mean(activations))
print("真实方差:", np.var(activations))


#

def variance(batch_size=20, momentum=0.3):
    batches = len(activations) // batch_size

    biased_var = []

    moving_var = []
    unbiased_var = []
    tmp = []
    m_var = 0
    s_idx = 0
    for i in range(batches):
        e_idx = i * batch_size + batch_size
        mini_batch = activations[s_idx:e_idx]

        bi_var = np.var(mini_batch)
        biased_var.append(bi_var)

        un_var = np.var(mini_batch) * len(mini_batch) / (len(mini_batch) - 1)
        tmp.append(un_var)
        unbiased_var.append(np.mean(tmp))

        m_var = momentum * m_var + (1 - momentum) * bi_var
        moving_var.append(m_var)
        s_idx = e_idx
    return unbiased_var, biased_var, moving_var


def plot():
    plt.subplots(1, 2, figsize=(9, 4))
    plt.subplot(1, 2, 1)
    momentum = 0.5
    unbiased_var, biased_var, moving_var = variance(batch_size=50, momentum=momentum)
    plt.hlines(np.var(activations), -1, len(unbiased_var) + 1, label='true var')
    # plt.plot(range(len(unbiased_var)), unbiased_var, label='unbiased var')
    plt.plot(range(len(unbiased_var)), biased_var, '--', label='mini-batch var in training')
    plt.plot(range(len(unbiased_var)), moving_var, label=f'moving average with momentum = {momentum}')
    plt.legend()

    plt.subplot(1, 2, 2)
    momentum = 0.9
    unbiased_var, biased_var, moving_var = variance(batch_size=50, momentum=momentum)
    plt.hlines(np.var(activations), -1, len(unbiased_var) + 1, label='true var')
    # plt.plot(range(len(unbiased_var)), unbiased_var, label='unbiased var')
    plt.plot(range(len(unbiased_var)), biased_var, '--', label='mini-batch var in training')
    plt.plot(range(len(unbiased_var)), moving_var, label=f'moving average with momentum = {momentum}')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot()
