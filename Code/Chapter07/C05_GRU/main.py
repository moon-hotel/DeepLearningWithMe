"""
文件名: Code/Chapter07/C05_GRU/main.py
创建时间: 2023/5/14 4:07 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn

torch.manual_seed(5)


def test_GRU():
    batch_size = 2
    time_step = 3
    input_size = 4
    hidden_size = 5
    x = torch.rand([batch_size, time_step, input_size])  # [batch_size, time_step, input_size]
    gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
    output, hn = gru(x)
    print(output)
    print(hn)



if __name__ == '__main__':
    test_GRU()
