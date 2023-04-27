"""
文件名: Code/Chapter06/C04_LN/layer_normalization.py
创建时间: 2023/4/17 8:35 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    def __init__(self,
                 normalized_shape=None,
                 dim=-1,
                 eps=1e-5):
        """
        :param normalized_shape: $\gamma$ 和 $\beta$的维度信息
        :param dim: 默认-1，即对最后一个维度进行标准化，即RNN
                    也可通过list指定相应维度，例如CNN中
        :param eps:
        """
        super(LayerNormalization, self).__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, X):
        mean = torch.mean(X, dim=self.dim, keepdim=True)
        # print(f"mena = {mean}")
        # 在指定的dim上做均值
        var = torch.mean((X - mean) ** 2, dim=self.dim, keepdim=True)
        # print(f"var = {var}")
        # 在指定的dim上做方差
        X_hat = (X - mean) / torch.sqrt(var + self.eps)
        # print(f"X_hat = {X_hat}")
        Y = self.gamma * X_hat + self.beta
        # print(f"self.gamma = {self.gamma.shape}")
        return Y


if __name__ == '__main__':
    #
    batch, sentence_length, embedding_dim = 2, 4, 3
    embedding = torch.tensor([[[1, 1, 2],
                               [2, 0, 0],
                               [0, 0, 1.],
                               [0, 1, 0]],
                              [[1, 0, 1],
                               [1, 1, 2],
                               [1, 0, 2],
                               [0, 2, 2]]])
    layer_norm = nn.LayerNorm(embedding_dim)
    my_layer_norm = LayerNormalization(embedding_dim)
    print(layer_norm(embedding))
    # tensor([[[-0.7071, -0.7071,  1.4142],
    #          [ 1.4142, -0.7071, -0.7071],
    #          [-0.7071, -0.7071,  1.4142],
    #          [-0.7071,  1.4142, -0.7071]],
    #
    #         [[ 0.7071, -1.4142,  0.7071],
    #          [-0.7071, -0.7071,  1.4142],
    #          [ 0.0000, -1.2247,  1.2247],
    #          [-1.4142,  0.7071,  0.7071]]], grad_fn=<NativeLayerNormBackward0>)
    print(my_layer_norm(embedding))
    # tensor([[[-0.7071, -0.7071,  1.4142],
    #          [ 1.4142, -0.7071, -0.7071],
    #          [-0.7071, -0.7071,  1.4142],
    #          [-0.7071,  1.4142, -0.7071]],
    #
    #         [[ 0.7071, -1.4142,  0.7071],
    #          [-0.7071, -0.7071,  1.4142],
    #          [ 0.0000, -1.2247,  1.2247],
    #          [-1.4142,  0.7071,  0.7071]]], grad_fn=<AddBackward0>)

    #
    print("IN CNN ===============================")
    N, C, H, W = 2, 3, 5, 5
    embedding = torch.randn(N, C, H, W)
    layer_norm = nn.LayerNorm([C, H, W])
    print(layer_norm(embedding))

    my_layer_norm = LayerNormalization([C, H, W], dim=[1, 2, 3])
    print(my_layer_norm(embedding))
