"""
文件名: Code/Chapter07/C04_LSTM/main.py
创建时间: 2023/5/12 21:48 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn

torch.manual_seed(5)


def test_LSTM():
    batch_size = 2
    time_step = 3
    input_size = 4
    hidden_size = 5
    x = torch.rand([batch_size, time_step, input_size])  # [batch_size, time_step, input_size]
    lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
    output, (hn, cn) = lstm(x)
    print(output)  # [batch_size, time_step, hidden_size]
    print(hn)  # [num_layer, batch_size, hidden_size]
    print(cn)  # [num_layer, batch_size, hidden_size]


if __name__ == '__main__':
    test_LSTM()
