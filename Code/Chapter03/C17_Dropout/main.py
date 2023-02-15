"""
文件名: Code/Chapter03/C17_Dropout/main.py
创建时间: 2023/2/14 7:36 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch._refs as refs
import torch
import torch.nn as nn


def dropout(a, drop_pro=0.5, training=True):
    """
    参考：torch/_refs/nn/functional/__init__.py
    :param a:
    :param drop_pro: 丢弃率
    :param training: 是否处于训练状态
    :return:
    """
    if not training:
        return a
    assert 0 <= drop_pro <= 1
    if drop_pro == 1:
        return refs.zeros_like(a)
    if drop_pro == 0:
        return a
    keep_pro = 1 - drop_pro
    scale = 1 / keep_pro
    mask = refs.uniform(a.shape, low=0.0, high=1.0, dtype=torch.float32, device=a.device) < keep_pro
    return refs.mul(refs.mul(a, mask), scale)


class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super(MyDropout, self).__init__()
        self.p = p

    def forward(self, x):
        return dropout(x, drop_pro=self.p, training=self.training)


if __name__ == '__main__':
    a = torch.randn([2, 10])
    op_dropout = MyDropout(p=0.2)
    print(op_dropout(a))

# tensor([[-1.4843, -1.0103, -0.0000,  0.7997,  0.9650,  0.4117, -0.6568, -0.4334,
#           1.5951, -0.9222],
#         [-1.2151,  2.4469, -1.9339, -0.6010, -0.0000,  0.0342, -1.1552,  0.0000,
#           0.1395,  1.7941]])
