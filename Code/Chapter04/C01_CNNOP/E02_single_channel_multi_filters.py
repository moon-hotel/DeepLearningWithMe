"""
文件名: Code/Chapter04/C01_CNNOP/E02_single_channel_multi_filters.py
创建时间: 2023/2/20 7:22 下午
"""
import torch.nn.functional as F
import torch

filters = torch.tensor([[[[0, 0, 1],
                          [0, 1, 1],
                          [1, 0, -1]]],
                        [[[1, 0, 0],
                          [0, 0, 1],
                          [1, 0, 1]]]])  # [2,1,3,3] [ [filter_nums/output_channels,input_channels,high,width]
inputs = torch.tensor([1, 2, 0, -1, 0,  # [batch_size,in_channels,high,width]
                       -1, 1, 0, 0, 1,
                       2, -1, 2, 1, 0,
                       -1, 1, 0, 0, 0,
                       2, 1, -1, 0, 0]).reshape([1, 1, 5, 5])

bias = torch.tensor([1, 2])
result = F.conv2d(inputs, filters, bias=bias, stride=1, padding=0)

print("输入数据为:\n", inputs)
print("输入数据的形状为:\n", inputs.shape)
print("卷积核的形状：[filter_nums/output_channels,input_channels,high,width]  ==>\n", filters.shape)
print("卷积后的结果:\n", result)
print("结果的形状:\n", result.shape)

# 输入数据为:
#  tensor([[[[ 1,  2,  0, -1,  0],
#           [-1,  1,  0,  0,  1],
#           [ 2, -1,  2,  1,  0],
#           [-1,  1,  0,  0,  0],
#           [ 2,  1, -1,  0,  0]]]])
# 输入数据的形状为:
#  torch.Size([1, 1, 5, 5])
# 卷积核的形状：[filter_nums/output_channels,input_channels,high,width]  ==>
#  torch.Size([2, 1, 3, 3])
# 卷积后的结果:
#  tensor([[[[ 2, -2,  4],
#           [ 1,  5,  3],
#           [ 7,  3,  0]],
#
#          [[ 7,  4,  5],
#           [ 2,  5,  2],
#           [ 5,  2,  3]]]])
# 结果的形状:
#  torch.Size([1, 2, 3, 3])
