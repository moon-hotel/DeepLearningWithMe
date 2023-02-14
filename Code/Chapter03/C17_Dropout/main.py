"""
文件名: Code/Chapter03/C17_Dropout/main.py
创建时间: 2023/2/14 7:36 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch._refs as refs
import torch


def dropout(a, p=0.5, training=True):
    """
    参考：torch/_refs/nn/functional/__init__.py
    :param a:
    :param p:
    :param training:
    :return:
    """
    if not training:
        return a
    assert p <= 1
    assert p >= 0
    if p == 1:
        return refs.zeros_like(a)
    if p == 0:
        return a
    p1m = 1 - p
    scale = 1 / p1m
    mask = refs.uniform(a.shape, low=0.0, high=1.0, dtype=torch.float32, device=a.device) < p1m
    return refs.mul(refs.mul(a, mask), scale)


if __name__ == '__main__':
    a = torch.randn([2, 10])
    y = dropout(a, p=0.4)
    print(y)
