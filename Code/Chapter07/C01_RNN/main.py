"""
文件名: Code/Chapter07/C01_RNN/main.py
创建时间: 2023/5/4 3:48 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)


def test_RNN():
    batch_size = 2
    time_step = 3
    input_size = 4
    hidden_size = 5
    x = torch.rand([batch_size, time_step, input_size])  # [batch_size, time_step, input_size]
    rnn = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True)
    output, hn = rnn(x)
    print(output.shape)
    print(output)
    print(hn)


def RNN_compute():
    X = np.array([[[0.4, 0.2, 0.5, 0.1],
                   [0.1, 0.3, 0.2, 0.0],
                   [0.0, 0.2, 0.4, 0.2]]])
    h0 = np.array([[0.0, 0.0]])
    U = np.array([[0.2, 0.5],
                  [0.1, 0.1],
                  [0.0, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.1, 0.1],
                  [0.0, 0.2]])
    b = np.array([[0.5, 0.5]])
    for i in range(X.shape[1]):
        print(f"time_step = {i + 1}")
        tmp = np.matmul(X[:, i], U) + np.matmul(h0, W) + b
        print(f"非线性变换前的结果为:\n {tmp}")
        h0 = np.tanh(tmp)
        print(f"线性变换后的结果h{i + 1}为:\n {h0}")


if __name__ == '__main__':
    test_RNN()
    RNN_compute()
