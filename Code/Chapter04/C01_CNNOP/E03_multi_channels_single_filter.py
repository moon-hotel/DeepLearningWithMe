"""
文件名: Code/Chapter04/C01_CNNOP/E03_multi_channels_single_filter.py
创建时间: 2023/2/20 7:37 下午
"""
import torch.nn.functional as F

import torch

inputs = torch.tensor([0, 2, 0, 1, 0, 0, 2, 0, 1, 1, 2, 1,
                       2, 0, 0, 1, 0, 0, 1, 0, -1, 1, 1, 0, 1,
                       0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0,
                       0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0,
                       1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
                       1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]).reshape([1, 3, 5, 5])

filters = torch.tensor([2, 0, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 0, 0, 1, 1,
                        0, 0, 0, 1, 1, 1, 1, 1, 1, 0]).reshape([1, 3, 3, 3])
# [1,3,3,3] [ [filter_nums/output_channels,input_channels,high,width]

bias = torch.tensor([1])
result = F.conv2d(inputs, filters, bias=bias, stride=1, padding=0)

print("输入数据为:\n", inputs)
print("输入数据的形状为:\n", inputs.shape)
print("卷积核的形状：[filter_nums/output_channels,input_channels,high,width]  ==>\n", filters.shape)
print("卷积后的结果:\n", result)
print("结果的形状:\n", result.shape)
