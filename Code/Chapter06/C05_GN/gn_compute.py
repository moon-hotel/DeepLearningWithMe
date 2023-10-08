"""
文件名: Code/Chapter06/C05_GN/gn_compute.py
创建时间: 2023/10/8 8:08 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn

if __name__ == '__main__':
    x = torch.tensor([0, 2, 0, 1, 0, 0, 2, 0, 1, 1, 2, 1, 2, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1,
                      0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0,
                      1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,  #
                      1, 2, 1, 1, 1, 0, 3, 1, 1, 1, 2, 0, 2, 2, 2, 1, 1, 2, 0, 0, 1, 0, 1, 0, 0,
                      0, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 0, 2, 1, 0, 1, 1, 2, 0, 0, 0, 1, 1, 0, 0,
                      0, 1, 1, 0, 0, 0, 1, 0, 2, 1, 1, 2, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0.,
                      1, 2, 1, 1, 1, 0, 3, 1, 1, 1, 2, 0, 2, 2, 2, 1, 1, 2, 0, 0, 1, 0, 1, 0, 0,
                      0, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 0, 2, 1, 0, 1, 1, 2, 0, 0, 0, 1, 1, 0, 0])
    origin_x = x.reshape([2, 4, 5, 5])
    num_groups = 2
    num_channels = 4
    assert num_channels % num_groups == 0
    x = origin_x.reshape([2, num_groups, -1, 5, 5]) # 分组
    print(f"手动实现：")
    mean = torch.mean(x, dim=[2, 3, 4], keepdim=True)  # 在每个样本的分组上做均值,
    print(f"mean = {mean}")
    print(f"mean shape = {mean.shape}")
    var = torch.mean((x - mean) ** 2, dim=[2, 3, 4], keepdim=True)  # 在每个样本的分组上求方差
    print(f"var = {var}")
    print(f"var shape = {var.shape}")
    X_hat = (x - mean) / torch.sqrt(var + 1e-5)
    X_hat = X_hat.reshape(-1, num_channels, 5, 5)
    print(f"X_hat = {X_hat}")
    print(f"X_hat shape = {X_hat.shape}")

    gamma = torch.ones([1, num_channels, 1, 1])
    beta = torch.zeros([1, num_channels, 1, 1])
    scaled_result = (X_hat * gamma + beta)
    print(f"scaled: {scaled_result}")
    print(f"scaled.shape: {scaled_result.shape}")
    #
    print(f"PyTorch实现：")
    gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    print(gn(origin_x))
    print(gn.weight.shape)

    ln = nn.LayerNorm([4, 5, 5])
    print(ln(origin_x))
    print(ln.weight.shape)
