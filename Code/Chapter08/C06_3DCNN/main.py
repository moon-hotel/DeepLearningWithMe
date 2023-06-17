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
import torch.nn.functional as F


def OP_3DCNNC():
    batch_size = 1
    in_channels = 3  # 输入通道数
    frame_len = 5  # 样本在时间维度的长度
    height = 32
    width = 32

    kernel_size = (2, 3, 3)  # (depth, h, w), 或者 kernel_size = 1
    stride = (1, 1, 1)  # 或者 stride = 1
    padding = (0, 1, 1)  # 分别对 depth, h, w这3个维度进行padding
    out_channels = 3

    m = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    input = torch.randn(batch_size, in_channels, frame_len, height, width)
    output = m(input)
    print(output.shape)  # [batch_size, out_channels, frame_len_out, h_out, w_out] torch.Size([1, 3, 4, 32, 32])


if __name__ == '__main__':
    # OP_3DCNNC()
    input = torch.tensor([[[[37, 26, 1, 45, 9],
                            [15, 19, 46, 13, 47],
                            [32, 16, 47, 32, 34],
                            [12, 26, 19, 15, 45],
                            [18, 4, 43, 40, 27]],
                           [[43, 40, 24, 29, 5],
                            [39, 17, 21, 17, 1],
                            [49, 47, 24, 38, 6],
                            [11, 40, 28, 41, 49],
                            [43, 14, 44, 8, 30]]],
                          [[[19, 8, 16, 22, 13],
                            [46, 32, 5, 29, 18],
                            [18, 42, 40, 49, 18],
                            [46, 30, 49, 9, 20],
                            [40, 39, 42, 8, 46]],
                           [[5, 19, 36, 41, 39],
                            [20, 24, 29, 20, 26],
                            [38, 1, 46, 10, 15],
                            [26, 35, 7, 47, 5],
                            [7, 12, 1, 1, 17]]],
                          [[[3, 42, 25, 44, 32],
                            [23, 29, 36, 15, 42],
                            [23, 23, 25, 7, 26],
                            [13, 42, 40, 23, 16],
                            [16, 34, 19, 26, 39]],
                           [[35, 48, 17, 33, 39],
                            [28, 16, 22, 1, 40],
                            [44, 23, 49, 7, 3],
                            [29, 39, 28, 23, 6],
                            [15, 20, 37, 41, 41]]]])


# nn.MaxPool3d

#