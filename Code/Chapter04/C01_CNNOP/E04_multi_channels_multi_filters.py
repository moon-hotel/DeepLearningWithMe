"""
文件名: Code/Chapter04/C01_CNNOP/E04_multi_channels_multi_filters.py
创建时间: 2023/2/20 7:37 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
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
                        0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
                        0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
                        0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).reshape([2, 3, 3, 3])
# [1,3,3,3] [ [filter_nums/output_channels,input_channels,high,width]

bias = torch.tensor([1, -3])
result = F.conv2d(inputs, filters, bias=bias, stride=1, padding=0)

print("输入数据为:\n", inputs)
print("输入数据的形状为:\n", inputs.shape)
print("卷积核的形状：[filter_nums/output_channels,input_channels,high,width]  ==>\n", filters.shape)
print("卷积后的结果:\n", result)
print("结果的形状:\n", result.shape)
