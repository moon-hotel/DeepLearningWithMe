"""
文件名: Code/Chapter04/C02_PaddingPooling/pooling.py
创建时间: 2023/2/25 3:55 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn

if __name__ == '__main__':
    inputs = torch.randn([5, 3, 32, 32], dtype=torch.float32)  # [batch_size,in_channels,high,width]
    net = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=10,
                                  kernel_size=3, stride=1, padding=1),  # width: 32/1 = 32
                        nn.MaxPool2d(kernel_size=2, stride=2, ), # (32-2+1) /2  = 16
                        nn.AvgPool2d(kernel_size=2, stride=1) # (16-2+1)/1 = 15
                        )
    result = net(inputs)
    print("输入数据的形状为: ", inputs.shape)
    print("池化后结果的形状:", result.shape)
