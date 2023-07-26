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
    result = model.predict(['小米 生态 链出 新品 ， 智能 聪明 ： 有 了 它 ， 老婆 都 变懒 了',
                            '哪些 瞬间 是 NBA 球员 回想起来 最 自豪 的 和 最 懊恼 的 ？'], k=2)

    logging.info(result)


if __name__ == '__main__':
    config = ModelConfig()
    train(config)
    inference(config)