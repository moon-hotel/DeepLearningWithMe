"""
文件名: Code/utils/tools.py
创建时间: 2023/4/12 8:27 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch


def get_gpus(num=None):
    """
    如果num为list，则返回list中对应的GPU；
    如果num为整数，则返回前num个gpu
    如果没有GPU则返回CPU
    :param num:
    :return:
    """
    gpu_nums = torch.cuda.device_count()
    if isinstance(num, list):
        devices = [torch.device(f'cuda:{i}')
                   for i in num if i < gpu_nums]
    else:
        devices = [torch.device(f'cuda:{i}')
                   for i in range(gpu_nums)][:num]
    return devices if devices else [torch.device('cpu')]
