import torch.nn.functional as F
import torch.nn as nn
import torch

filters = torch.tensor([[[[2, 0, 0],
                          [1, 0, 1],
                          [0, 3, 0]],
                         [[1, 0, 1],
                          [0, 0, 0],
                          [1, 1, 0]],
                         [[0, 0, 1],
                          [1, 1, 1],
                          [1, 1, 0]]]])  # [1,3,3,3] [ [filter_nums/output_channels,input_channels,high,width]

inputs = torch.tensor([0, 2, 0, 1, 0, 0, 2, 0, 1, 1, 2, 1, 2, 0, 0, 1, 0, 0, 1, 0, -1, 1, 1, 0, 1,
                       0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0,
                       # [batch_size,in_channels,high,width]
                       1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]).reshape([1, 3, 5, 5])

bias = torch.tensor([1])
op = nn.Conv2d(in_channels=)
result = F.conv2d(inputs, filters, bias=bias, stride=1, padding=0)

print("输入数据为：", inputs)
print("输入数据的形状为：", inputs.shape)
print("卷积核的形状：[filter_nums/output_channels,input_channels,high,width]  ==>", filters.shape)
print("卷积后的结果：", result)
print("结果的形状：", result.shape)
