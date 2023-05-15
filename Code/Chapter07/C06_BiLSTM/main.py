"""
文件名: Code/Chapter07/C06_BiLSTM/main.py
创建时间: 2023/5/15 7:25 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn

torch.manual_seed(5)


def test_GRU():
    batch_size = 3
    time_step = 5
    input_size = 6
    hidden_size = 4
    x = torch.rand([batch_size, time_step, input_size])  # [batch_size, time_step, input_size]
    lstm = nn.LSTM(input_size, hidden_size, num_layers=1,
                   batch_first=True, bidirectional=True)
    output, (hn, cn) = lstm(x)
    print(output.shape)  # [batch_size, time_step, 2*hidden_size]
    print(output)
    print(hn.shape)  # [bidirectional*2, batch_size, hidden_size]
    print(hn)



if __name__ == '__main__':
    test_GRU()
