import torch.nn.functional as F
import torch

filter = torch.tensor([[[[0, 0, 1],
                         [0, 1, 1],
                         [1, 0, -1]]],
                       [[[1, 0, 0],
                         [0, 0, 1],
                         [1, 0, 1]]]])
inputs = torch.tensor([1, 2, 0, -1, 0, 1, 1, 0, 0, 1, 2, -1, 2, 1, 0,
                       -1, 1, 0, 0, 0, 2, 1, -1, 0, 0]).reshape([1, 1, 5, 5])
bias = torch.tensor([1, 2])
print(F.conv2d(inputs, filter, padding=0, bias=bias))

filters = torch.tensor([[[[2, 0, 0],
                          [1, 0, 1],
                          [0, 3, 0]],
                         [[1, 0, 1],
                          [0, 0, 0],
                          [1, 1, 0]],
                         [[0, 0, 1],
                          [1, 1, 1],
                          [1, 1, 0]]],
                        [[[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]],
                         [[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]],
                         [[1, 0, 1],
                          [0, 1, 0],
                          [1, 0, 1]]]
                        ])  # [2,3,3,3] [ [filter_nums/output_channels,input_channels,high,width]
print("卷积核的形状：[filter_nums/output_channels,input_channels,high,width]", filters.shape)
inputs = torch.tensor([0, 2, 0, 1, 0, 0, 2, 0, 1, 1, 2, 1, 2, 0, 0, 1, 0, 0, 1, 0, -1, 1, 1, 0, 1,
                       0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0,
                       # [batch_size,in_channels,high,width]
                       1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]).reshape([1, 3, 5, 5])

bias = torch.tensor([1, -3])
result = F.conv2d(inputs, filters, padding=0, bias=bias)
print("卷积后的结果：", result)
print("结果的形状：", result.shape)
