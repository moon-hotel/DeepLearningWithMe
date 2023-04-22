"""
文件名: Code/Chapter06/C02_BN/batch_normalization.py
创建时间: 2023/4/7 2:12 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn


class BatchNormalization(nn.Module):
    def __init__(self,
                 num_features=None,
                 num_dims=4,
                 momentum=0.1,
                 eps=1e-5):
        super(BatchNormalization, self).__init__()
        shape = [1, num_features]
        if num_dims == 4:
            shape = [1, num_features, 1, 1]

        self.momentum = momentum
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.register_buffer('moving_mean', torch.zeros(shape))
        self.register_buffer('moving_var', torch.zeros(shape))


    def forward(self, inputs):
        X = inputs
        if len(X.shape) not in (2, 4):
            raise ValueError("only support dense or 2dconv")

        if self.training:
            if len(X.shape) == 2:  # 全连接
                mean = torch.mean(X, dim=0)  #
                var = torch.mean((X - mean) ** 2, dim=0)
            else:  # 2d卷积
                mean = torch.mean(X, dim=[0, 2, 3], keepdim=True)
                # 在通道上做均值, [1,self.num_features,1,1]
                var = torch.mean((X - mean) ** 2, dim=[0, 2, 3], keepdim=True)
                # 在通道上求方差, [1,self.num_features,1,1]
            X_hat = (X - mean) / torch.sqrt(var + self.eps)
            self.moving_mean = self.momentum * self.moving_mean + (1.0 - self.momentum) * mean
            self.moving_var = self.momentum * self.moving_var + (1.0 - self.momentum) * var
        else:
            X_hat = (X - self.moving_mean) / torch.sqrt(self.moving_var + self.eps)
        Y = self.gamma * X_hat + self.beta
        return Y


if __name__ == '__main__':
    x = torch.randint(0, 10, (1, 2, 4, 4), dtype=torch.float32)
    mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)  # 在通道上做均值, [1,self.num_features,1,1]
    var = torch.mean((x - mean) ** 2, dim=[0, 2, 3], keepdim=True)  # 在通道上求方差, [1,self.num_features,1,1]
    X_hat = (x - mean) / torch.sqrt(var + 1e-5)
    print(X_hat)

    bn = BatchNormalization(num_features=2, num_dims=4)
    print(bn(x))

    bn = nn.BatchNorm2d(num_features=2)
    print(bn(x))
