"""
文件名: Code/utils/tools.py
创建时间: 2023/4/12 8:27 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import logging
import time
import os
import numpy as np
from datetime import datetime
import pandas as pd
import re


def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')  # 匹配中文字符的正则表达式
    return bool(re.search(pattern, text))


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


def process_cache(unique_key=None):
    """
    数据预处理结果缓存修饰器
    :param : unique_key
    :return:
    """
    if unique_key is None:
        raise ValueError(
            "unique_key 不能为空, 请指定相关数据集构造类的成员变量，如['top_k', 'cut_words', 'max_sen_len']")

    def decorating_function(func):
        def wrapper(*args, **kwargs):
            logging.info(f" ## 索引预处理缓存文件的参数为：{unique_key}")
            obj = args[0]  # 获取类对象，因为data_process(self, file_path=None)中的第1个参数为self
            file_path = kwargs['file_path']
            file_dir = f"{os.sep}".join(file_path.split(os.sep)[:-1])
            file_name = "".join(file_path.split(os.sep)[-1].split('.')[:-1])
            paras = f"cache_{file_name}_"
            for k in unique_key:
                paras += f"{k}{obj.__dict__[k]}_"  # 遍历对象中的所有参数
            cache_path = os.path.join(file_dir, paras[:-1] + '.pt')
            start_time = time.time()
            if not os.path.exists(cache_path):
                logging.info(f"缓存文件 {cache_path} 不存在，重新处理并缓存！")
                data = func(*args, **kwargs)
                with open(cache_path, 'wb') as f:
                    torch.save(data, f)
            else:
                logging.info(f"缓存文件 {cache_path} 存在，直接载入缓存文件！")
                with open(cache_path, 'rb') as f:
                    data = torch.load(f)
            end_time = time.time()
            logging.info(f"数据预处理一共耗时{(end_time - start_time):.3f}s")
            return data

        return wrapper

    return decorating_function


def string2timestamp(strings, T=48):
    """
    将字符串类型的时间转换成时间戳格式
    :param strings:
    :param T: 表示一天有多少个时间片
    :return:
    example:
    str = [b'2013070101', b'2013070102']
    如 2013070101表示 2013年7月1日第1个时间片
    print(string2timestamp(str))
    [Timestamp('2013-07-01 00:00:00'), Timestamp('2013-07-01 00:30:00')]
    """
    timestamps = []

    time_per_slot = 24.0 / T  # 每个时间片多少小时
    num_per_T = T // 24  # 每个小时有多少时间片
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:]) - 1
        timestamp = datetime(year, month, day, hour=int(slot * time_per_slot),
                             minute=(slot % num_per_T) * int(60.0 * time_per_slot))
        timestamps.append(pd.Timestamp(timestamp))
    return timestamps


def timestamp2vec(timestamps):
    """
    将字符串类型的时间换为表示星期几和工作日的向量
    :param timestamps:
    :return:
    exampel:
    [b'2018120505', b'2018120106']
    #[[0 0 1 0 0 0 0 1]  当天是星期三，且为工作日
     [0 0 0 0 0 1 0 0]]  当天是星期六，且为休息日

    """
    # tm_wday range [0, 6], Monday is 0
    vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]
    # 通过tm_wday方法得到当天是星期几
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)


class MinMaxNormalization(object):
    """
    特征缩放，公式如下：
        MinMax Normalization --> [-1, 1]
        x = (x - min) / (max - min).
        x = x * 2 - 1
    """

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        logging.info(f"MinMaxNormalization: min = {self._min}, max = {self._max}")

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X
