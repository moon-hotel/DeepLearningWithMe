"""
文件名: Code/Chapter07/C01_RNN.py
创建时间: 2023/5/4 3:48 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn

torch.manual_seed(42)


def test_RNN():
    batch_size = 2
    time_step = 3
    input_size = 4
    hidden_size = 5
    rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    x = torch.rand([batch_size, time_step, input_size])  # [batch_size, time_step, input_size]
    output, hn = rnn(x)
    print(output)
    print(hn)
    # print("sss", rnn.all_weights)


if __name__ == '__main__':
    test_RNN()
