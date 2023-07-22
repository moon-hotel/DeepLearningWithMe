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
    # path_to_model = os.path.join(DATA_HOME, 'Pretrained', 'fasttext', 'cc.zh.300.vec.gz')
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


def get_get_analogies():
    ft = fasttext.load_model(os.path.join(DATA_HOME, 'Pretrained', 'fasttext', 'cc.zh.300.bin'))
    logging.info('有凤来仪' in ft.words)
    logging.info(f"与东坡居士最相似的5个词为: {ft.get_nearest_neighbors('有凤来仪', k=5)}")
    logging.info(ft.get_analogies("柏林", "德国", "法国", k=5))

    # False
    # 与有凤来仪最相似的5个词为: [(0.457183, 'Viscosity'), (0.454175, 'viscosity'), (0.361536, 'thb'), (0.343013, 'kg/m2'), (0.335760, 'Dirham')]
    # [(0.743810, '巴黎'), (0.583832, '里昂'), (0.555544, '法國'), (0.547275, '斯特拉斯堡'), (0.536760, '坎城')]

    ft = fasttext.load_model(os.path.join(DATA_HOME, 'Pretrained', 'fasttext', 'cc.en.300.bin'))
    logging.info('accomodtion' in ft.words)
    logging.info(f"与accomodtion最相似的5个词为: {ft.get_nearest_neighbors('accomodation', k=5)}")
    logging.info(ft.get_analogies("berlin", "germany", "france", k=5))
    # False
    # 与accomodtion最相似的5个词为: [(0.858731, 'accomadation'), (0.828016, 'acommodation'), (0.822644, 'accommodation'), (0.821873, 'accomdation'), (0.793275, 'Accomodation')]
    # [(0.730373, 'paris'), (0.640853, 'france.'), (0.639331, 'avignon'), (0.631667, 'paris.'), (0.589559, 'montpellier')]


if __name__ == '__main__':
    # load_fasttext_model()
    get_get_analogies()
