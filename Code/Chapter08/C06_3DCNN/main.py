"""
文件名: Code/Chapter08/C06_3DCNN/main.py
创建时间: 2023/6/13 8:46 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
Tran D, Bourdev L, Fergus R, et al. Learning spatiotemporal features with 3d convolutional
networks[C]//Proceedings of the IEEE international conference on computer vision. 2015: 4489-4497.
"""

import torch.nn as nn
import torch

if __name__ == '__main__':
    batch_size = 2
    in_channels = 3  # 输入通道数
    frame_len = 10  # 样本在时间维度的长度
    height = 32
    width = 32

    kernel_size = (3, 3, 5)  # (depth, h, w), 或者 kernel_size = 1
    stride = (1, 1, 1)  # 或者 stride = 1
    padding = (0, 1, 2)  # 分别对 depth, h, w这3个维度进行padding
    out_channels = 4

    m = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    input = torch.randn(batch_size, in_channels, frame_len, height, width)
    output = m(input)
    print(output.shape)  # [batch_size, out_channels, frame_len_out, h_out,w_out]

#
