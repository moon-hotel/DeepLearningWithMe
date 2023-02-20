"""
文件名: Code/Chapter04/C01_CNNOP/E01_single_channel_single_filter.py
创建时间: 2023/2/19 8:41 下午
"""
import torch.nn.functional as F
import torch

filters = torch.tensor([[[[0, 0, 1],
                          [0, 1, 1],
                          [1, 0, -1]]]])  # [1,1,3,3] [ [filter_nums/output_channels,input_channels,high,width]

inputs = torch.tensor([1, 2, 0, -1, 0,  # [batch_size,in_channels,high,width]
                       -1, 1, 0, 0, 1,
                       2, -1, 2, 1, 0,
                       -1, 1, 0, 0, 0,
                       2, 1, -1, 0, 0]).reshape([1, 1, 5, 5])

bias = torch.tensor([1])
result = F.conv2d(inputs, filters, bias=bias, stride=1, padding=0)

print("输入数据为:\n", inputs)
print("输入数据的形状为:\n", inputs.shape)
print("卷积核的形状：[filter_nums/output_channels,input_channels,high,width]  ==>\n", filters.shape)
print("卷积后的结果:\n", result)
print("结果的形状:\n", result.shape)
