"""
文件名: Code/Chapter06/C05_GN/group_normalization.py
创建时间: 2023/10/7 8:15 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn


class GroupNormalization(nn.Module):
    def __init__(self,
                 num_groups,
                 num_channels,
                 eps=1e-5):
        """
        :param eps:
        """
        super(GroupNormalization, self).__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones([1, num_channels, 1, 1]))
        self.beta = nn.Parameter(torch.zeros([1, num_channels, 1, 1]))

    def forward(self, X):
        w, h = X.shape[-2:]
        X = X.reshape([-1, self.num_groups, self.num_channels // self.num_groups, w, h])
        mean = torch.mean(X, dim=[2, 3, 4], keepdim=True)  # 在每个样本的分组上做均值
        # [batch_sie,num_groups,1,1,1]
        var = torch.mean((X - mean) ** 2, dim=[2, 3, 4], keepdim=True)  # 在每个样本的分组上求方差
        # [batch_sie,num_groups,1,1,1]
        X_hat = (X - mean) / torch.sqrt(var + self.eps)
        # [batch_sie,num_groups,self.num_channels // self.num_groups,w,h]
        X_hat = X_hat.reshape(-1, self.num_channels, w, h)
        Y = self.gamma * X_hat + self.beta
        return Y


if __name__ == '__main__':
    x = torch.randn([2, 6, 5, 5])
    num_groups = 2
    num_channels = 6
    gn = GroupNormalization(num_groups, num_channels)
    y = gn(x)
    print(y[0][0][0])
    gn = nn.GroupNorm(num_groups, num_channels)
    y = gn(x)
    print(y[0][0][0])