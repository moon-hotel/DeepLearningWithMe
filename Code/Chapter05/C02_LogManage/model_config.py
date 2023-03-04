"""
文件名: Code/Chapter05/C02_LogManage/model_config.py
创建时间: 2023/3/4 8:57 上午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import json
import logging
from log_manage import logger_init


class ModelConfig(object):
    def __init__(self, batch_size=16,
                 learning_rate=3.5e-5,
                 num_labels=3,
                 epochs=5):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        self.epochs = epochs

        logging.info("\n ####  <----------------------->")
        for key, value in self.__dict__.items():
            logging.info(f"##  {key} = {value}")


if __name__ == '__main__':
    logger_init(log_file_name='monitor', log_level=logging.DEBUG,
                log_dir='./logs', only_file=False)
    config = ModelConfig()
