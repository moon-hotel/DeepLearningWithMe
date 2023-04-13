"""
文件名: Code/utils/tools.py
创建时间: 2023/4/12 4:27 下午
"""

import torch


def get_gpus(num=10):
    """
    如果有GPU则返回前num个gpu
    如果没有GPU则返回CPU
    :param num:
    :return:
    """

    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())][:num]
    return devices if devices else [torch.device('cpu')]
