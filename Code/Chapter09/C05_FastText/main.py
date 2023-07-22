"""
文件名: Code/Chapter09/C05_FastText/main.py
创建时间: 2023/7/22 10:31 上午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import logging
from gensim.models import KeyedVectors
import fasttext
from fasttext.util import reduce_model
import sys
import os

sys.path.append('../../')
from utils import DATA_HOME


def load_fasttext_model():
    path_to_model = os.path.join(DATA_HOME, 'Pretrained', 'fasttext', 'cc.zh.300.bin')
    # path_to_model = os.path.join(DATA_HOME, 'Pretrained', 'fasttext', 'cc.zh.300.vec')
    # model = KeyedVectors.load_word2vec_format(path_to_model, binary=False)
    ft = fasttext.load_model(path_to_model)
    logging.info(f"词向量的维度: {ft.get_dimension()}")
    logging.info(f"中国: {ft.get_word_vector('中国')}")
    logging.info(f"与中国最相似的5个词为: {ft.get_nearest_neighbors('中国', k=5)}")
    logging.info(ft.get_subwords("跟我一起学深度学习"))
    reduce_model(ft, 100)  # 降维
    logging.info(f"词向量的维度: {ft.get_dimension()}")
    path_to_model = os.path.join(DATA_HOME, 'Pretrained', 'fasttext', 'cc.zh.100.bin')
    ft.save_model(path_to_model)



if __name__ == '__main__':
    load_fasttext_model()
