"""
文件名: Code/Chapter04/C02_PaddingPooling/main.py
创建时间: 2023/2/21 8:45 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn

if __name__ == '__main__':
    inputs = torch.randn([5, 3, 32, 32], dtype=torch.float32)  # [batch_size,in_channels,high,width]
    cnn_op = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
    result = cnn_op(inputs)
    print("输入数据的形状为:\n", inputs.shape)
    print("结果的形状:\n", result.shape)  # width: 32/1

    cnn_op = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=2, padding=1)
    result = cnn_op(inputs)
    print("输入数据的形状为:\n", inputs.shape)
    print("结果的形状:\n", result.shape)  # width: (32+2-5+1)//2 = 15
