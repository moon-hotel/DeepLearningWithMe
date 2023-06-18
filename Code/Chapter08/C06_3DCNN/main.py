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
    stride = (1, 1, 1)  # 或者 stride = 1， 分别表示在depth, h, w这3个维度上进行移动
    padding = (0, 1, 1)  # 分别对 depth, h, w这3个维度进行padding
    out_channels = 3

    m = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    input = torch.randn(batch_size, in_channels, frame_len, height, width)
    output = m(input)
    print(output.shape)  # [batch_size, out_channels, frame_len_out, h_out, w_out] torch.Size([1, 3, 4, 32, 32])


def compute():
    input = torch.tensor([[[[[8, 2, 2, 1, 4],[9, 2, 6, 9, 4],[5, 9, 0, 6, 6],[1, 7, 1, 4, 9],[1, 2, 3, 7, 6]], # channel 1   frame 1
                            [[7, 0, 1, 6, 8], [2, 2, 5, 9, 0], [3, 2, 1, 8, 6], [5, 1, 6, 2, 5], [4, 9, 9, 4, 8]], #         frame 2
                            [[9, 7, 7, 6, 7], [0, 9, 7, 4, 9], [3, 5, 3, 3, 7], [5, 9, 7, 6, 7], [6, 9, 3, 2, 2]]],#         frame 3

                           [[[3, 3, 3, 7, 5], [3, 7, 7, 4, 3], [6, 6, 0, 7, 7], [5, 3, 2, 0, 8], [0, 1, 9, 4, 4]], # channel 2  frame 1
                            [[1, 0, 0, 8, 8], [8, 0, 5, 3, 3], [5, 5, 4, 0, 6], [7, 9, 1, 3, 6], [9, 2, 1, 6, 1]], #            frame 2
                            [[3, 5, 1, 9, 7], [1, 5, 5, 8, 0], [1, 5, 9, 2, 2], [5, 4, 8, 6, 7], [1, 6, 3, 2, 1]]]]]) #         frame 3
    # print(input.shape) # [batch_size, in_channels, frame_len, height, width] torch.Size([1,2,3, 5, 5])

    weight = torch.tensor([[[[[1, 1, 0], [0, 0, 0], [1, 1, 1]],
                             [[0, 1, 1], [1, 1, 0], [0, 1, 1]]],
                            [[[1, 0, 0], [0, 1, 0], [1, 1, 0]],
                             [[1, 0, 1], [0, 2, 1], [1, 0, 2]]]],

                           [[[[0, 1, 2], [1, 1, 0], [1, 1, 2]],
                             [[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
                            [[[1, 0, 2], [1, 0, 2], [0, 1, 2]],
                             [[2, 0, 2], [1, 1, 0], [0, 1, 0]]]]])
    # print(weight.shape) # [out_channels, in_channels, depth, height, width] torch.Size([2,2,2,3,3])
    bias = torch.tensor([-70, -80])
    result = F.conv3d(input, weight, bias)
    print(result)  # [batch_size, out_channels, frame_len_out, h_out, w_out] torch.Size([1, 2, 2, 3, 3])
    # tensor([[[[[  3,  14,  34],
    #            [ 22,  13,  27],
    #            [ 26,  22,  24]],
    #
    #           [[ 23,  32,  30],
    #            [ 61,  41,  31],
    #            [ 49,  30,  36]]],
    #
    #
    #          [[[ 21,  58,  77],
    #            [ 60,  58,  74],
    #            [ 56,  62,  88]],
    #
    #           [[ 34,  83, 100],
    #            [ 67,  76,  78],
    #            [ 72,  79,  92]]]]])


def max_pool():
    input = torch.tensor(
        [[[[[8, 2, 2, 1, 4], [9, 2, 6, 9, 4], [5, 9, 0, 6, 6], [1, 7, 1, 4, 9], [1, 2, 3, 7, 6]],  # channel 1   frame 1
           [[7, 0, 1, 6, 8], [2, 2, 5, 9, 0], [3, 2, 1, 8, 6], [5, 1, 6, 2, 5], [4, 9, 9, 4, 8]],  #             frame 2
           [[9, 7, 7, 6, 7], [0, 9, 7, 4, 9], [3, 5, 3, 3, 7], [5, 9, 7, 6, 7], [6, 9, 3, 2, 2]]], #             frame 3

          [[[3, 3, 3, 7, 5], [3, 7, 7, 4, 3], [6, 6, 0, 7, 7], [5, 3, 2, 0, 8], [0, 1, 9, 4, 4]],  # channel 2  frame 1
           [[1, 0, 0, 8, 8], [8, 0, 5, 3, 3], [5, 5, 4, 0, 6], [7, 9, 1, 3, 6], [9, 2, 1, 6, 1]],  #            frame 2
           [[3, 5, 1, 9, 7], [1, 5, 5, 8, 0], [1, 5, 9, 2, 2], [5, 4, 8, 6, 7], [1, 6, 3, 2, 1.]]]]]) #         frame 3
    # print(input.shape) # [batch_size, in_channels, frame_len, height, width] torch.Size([1,2,3, 5, 5])
    pool = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=1)
    # pool = nn.AvgPool3d(kernel_size=(2, 3, 3), stride=1)
    print(pool(input))  # shape:  [batch_size, out_channels, frame_len_out, h_out, w_out] torch.Size([1, 2, 2, 3, 3])



if __name__ == '__main__':
    # compute()
    # OP_3DCNNC()
    max_pool()
