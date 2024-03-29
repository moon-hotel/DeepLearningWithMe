"""
文件名: Code/Chapter06/C03_BN/bn_compute.py
创建时间: 2023/4/23 20:02 下午
"""

import torch
import torch.nn as nn

if __name__ == '__main__':
    x = torch.tensor([0, 2, 0, 1, 0, 0, 2, 0, 1, 1, 2, 1, 2, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1,
                      0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0,
                      1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,  #
                      1, 2, 1, 1, 1, 0, 3, 1, 1, 1, 2, 0, 2, 2, 2, 1, 1, 2, 0, 0, 1, 0, 1, 0, 0,
                      0, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 0, 2, 1, 0, 1, 1, 2, 0, 0, 0, 1, 1, 0, 0,
                      0, 1, 1, 0, 0, 0, 1, 0, 2, 1, 1, 2, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0.])
    x = x.reshape([2, 3, 5, 5])
    shape = [1, 3, 1, 1]
    # print(x)
    print(f"手动实现：")
    mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)  # 在通道上做均值, [1,self.num_features,1,1]
    print(f"mean = {mean}")
    var = torch.mean((x - mean) ** 2, dim=[0, 2, 3], keepdim=True)  # 在通道上求方差, [1,self.num_features,1,1]
    print(f"var = {var}")
    X_hat = (x - mean) / torch.sqrt(var + 1e-5)
    gamma = torch.reshape(torch.tensor([0.5, 0.1, 0.2]), shape)
    beta = torch.reshape(torch.tensor([0, 0.1, 0]), shape)
    print(X_hat*gamma + beta)
    print(f"PyTorch实现：")
    bn = nn.BatchNorm2d(num_features=3)
    bn.weight = nn.Parameter(gamma)
    bn.bias = nn.Parameter(beta)
    print(bn(x))
