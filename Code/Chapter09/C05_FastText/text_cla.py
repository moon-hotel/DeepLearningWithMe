"""
文件名: Code/Chapter09/C05_FastText/text_cla.py
创建时间: 2023/7/23 09:11 上午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import fasttext
import sys
import os
import logging

import torch.nn

sys.path.append("../../")
from utils import DATA_HOME
from utils import logger_init

logger_init(log_file_name='log', log_level=logging.INFO, log_dir='log')
DATA_DIR = os.path.join(DATA_HOME, 'toutiao')
FILE_PATH = [os.path.join(DATA_DIR, 'toutiao_train_fasttext.txt'),
             os.path.join(DATA_DIR, 'toutiao_val_fasttext.txt'),
             os.path.join(DATA_DIR, 'toutiao_test_fasttext.txt')]


class ModelConfig(object):
    def __init__(self):
        self.epochs = 5
        self.word_ngrams = 3
        self.learning_rate = 0.15
        self.vector_size = 100
        self.min_count = 1
        self.minn = 3
        self.maxn = 6
        self.label = '_!_'
        self.model_save_path = 'fasttext.bin'
        self.data_path = FILE_PATH
        logging.info("### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def train(config):
    model = fasttext.train_supervised(input=config.data_path[0], lr=config.learning_rate,
                                      dim=config.vector_size, epoch=config.epochs, minCount=config.min_count,
                                      minn=config.minn, maxn=config.maxn, wordNgrams=config.word_ngrams,
                                      label=config.label)
    logging.info(model.test(config.data_path[1]))
    model.save_model("fasttext.bin")


def inference(config):
    model = fasttext.load_model(config.model_save_path)
    text = ['小米生态链出新品，智能聪明：有了它，老婆都变懒了',
            '哪些瞬间是NBA球员回想起来最自豪的和最懊恼的？']
    result = model.predict(text, k=2)

    logging.info(result)


if __name__ == '__main__':
    config = ModelConfig()
    train(config)
    inference(config)
