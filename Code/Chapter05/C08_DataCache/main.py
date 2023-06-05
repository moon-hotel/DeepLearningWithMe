"""
文件名: Code/Chapter05/C08_DataCache/main.py
创建时间: 2023/6/4 5:42 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import logging

import sys
import time

sys.path.append("../../")
from utils import process_cache


class LoadData(object):
    FILE_PATH = './text_train.txt'

    def __init__(self):
        self.max_len = 5
        self.batch_size = 2

    @process_cache(['max_len'])
    def data_process(self, file_path=None):
        time.sleep(10)
        logging.info("正在进行预处理数据……")
        x = torch.randn((10, 5))
        y = torch.randint(2, [10])
        data = {"x": x, "y": y}
        return data

    def load_train_val_test_data(self):
        data = self.data_process(file_path=self.FILE_PATH)
        x, y = data['x'], data['y']
        data_iter = TensorDataset(x, y)
        data_iter = DataLoader(data_iter, batch_size=self.batch_size)
        return data_iter


if __name__ == '__main__':
    data = LoadData()
    data_iter = data.load_train_val_test_data()
    for x, y in data_iter:
        print(x.shape, y.shape)
