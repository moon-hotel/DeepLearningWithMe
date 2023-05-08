"""
文件名: Code/Chapter07/C03_RNNNewsCla/ont_hot.py
创建时间: 2023/5/7 2:57 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch

torch.random.manual_seed(2020)
if __name__ == '__main__':
    x = torch.randint(0, 10, [2, 3], dtype=torch.long)
    x_one_hot = torch.nn.functional.one_hot(x, num_classes=10)
    print(x)
    print(x_one_hot)
