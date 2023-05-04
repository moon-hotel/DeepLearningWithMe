"""
文件名: Code/Chapter06/C04_LN/ln_compute.py
创建时间: 2023/4/23 8:08 下午
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
                      0, 1, 1, 0, 0, 0, 1, 0, 2, 1, 1, 2, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0.])
    x = x.reshape([2, 3, 5, 5])
    # print(x)
    print(f"手动实现：")
    mean = torch.mean(x, dim=[1, 2, 3], keepdim=True)  # 在每个样本上做均值,
    print(f"mean = {mean}")
    var = torch.mean((x - mean) ** 2, dim=[1, 2, 3], keepdim=True)  # 在每个样本上求方差
    print(f"var = {var}")
    X_hat = (x - mean) / torch.sqrt(var + 1e-5)
    print(f"X_hat = {X_hat}")
    gamma = torch.ones([3, 5, 5])
    beta = torch.zeros([3, 5, 5])
    print(f"scaled: {X_hat * gamma + beta}")

    print(f"PyTorch实现：")
    ln = nn.LayerNorm([3, 5, 5])
    ln.weight = nn.Parameter(gamma)
    ln.bias = nn.Parameter(beta)
    print(ln(x))
