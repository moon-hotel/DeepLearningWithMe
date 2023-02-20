"""
文件名: Code/Chapter04/C01_CNNOP/E05_pooling.py
创建时间: 2023/2/20 8:03 下午
"""
import torch
import torch.nn.functional as F

inputs = torch.tensor([5, 2, 0, 1, 0, 0, 2, 3, 0, 7, 2, 3, 2, 2, 1, 1, 6, 4, 8, 1, 2, 7, 1, 5, 9,
                       4, 2, 0, 1, 0, 2, 7, 1, 3, 6, 2, 4, 2, 2, 3, 2, 6, 9, 8, 0, 10, 7, 2, 5, 7], dtype=torch.float32)
inputs = inputs.reshape([1, 2, 5, 5])
result = F.max_pool2d(inputs, kernel_size=3, stride=1)
print(result)
