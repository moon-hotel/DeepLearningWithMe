"""
文件名: Code/Chapter09/C01_Word2Vec/main.py
创建时间: 2023/7/12 20:31 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from gensim.models import KeyedVectors
import logging
import sys
import os

sys.path.append('../../')
from utils import DATA_HOME


def load_third_part_wv_en():
    path_to_model = os.path.join(DATA_HOME, 'Pretrained', 'GoogleNews-vectors-negative300.bin.gz')
    model = KeyedVectors.load_word2vec_format(path_to_model, binary=True)
    vec_king = model['king']
    vec_queen = model['queen']
    logging.info(f"vec_king: {vec_king}")  # [ 0.125976562e  0.0297851562e  0.00860595703e 0.139648438e....
    logging.info(f"vec_queen: {vec_queen}")  # [ 0.00524902 -0.14355469 -0.06933594  0.12353516  0.13183594....
    # ① 计算两个词之间的相似度(cosine)，结果越大越相似
    sim1 = model.similarity('king', 'queen')
    sim2 = model.similarity('king', 'soldiers')
    logging.info(f"king 和 queen 之间的相似度为: {sim1}")  # 0.6510956883430481
    logging.info(f"king 和 soldiers 之间的相似度为: {sim2}")  # 0.12567302584648132

    # ② 计算两个词之间的距离 1 - self.similarity(w1, w2)
    dist = model.distance('king', 'soldiers')
    logging.info(f"king 和 soldiers 之间的距离为: {dist}")  # 0.8743269741535187

    # ③ 找最相似的K个词
    sim_words = model.most_similar(['king'], topn=3)
    logging.info(f"与king最相似的前3个词为:{sim_words}")
    # [('kings', 0.7138045430183411), ('queen', 0.6510956883430481), ('monarch', 0.6413194537162781)]

    # ④ 找出同时与给定多个词最相似的K个词
    sim_words = model.most_similar(['king', 'queen'], topn=3)
    logging.info(f"同时与king和queen最相似的前3个词为:{sim_words}")
    # [('monarch', 0.7042067050933838), ('kings', 0.6780861616134644), ('princess', 0.6731551885604858)]

    # ⑤ 找出与positive最相似，但排查与negative相似的词

    sim_words = model.most_similar(positive=['apple'], topn=3)
    logging.info(f"与apple最相似的前3个词为:{sim_words}")
    # [('apples', 0.720359742641449), ('pear', 0.6450697183609009), ('fruit', 0.6410146355628967)]
    sim_words = model.most_similar(positive=['apple'], negative=['fruit'], topn=6)
    logging.info(f"与apple最相似，但是与fruit不相似的前6个词为:{sim_words}")
    # [('Apple', 0.33312755823135376), ('Appleâ_€_™', 0.3215164244174957), ('Ipod', 0.31791260838508606)]

    # ⑥ 找出其中与其它词差异最大的词
    does_match = model.doesnt_match(['king', 'queen', 'soldiers'])
    logging.info(f"king,queen,soldiers中与其它两个词最不相似的前为:{does_match}")


def load_third_part_wv_zh():
    path_to_model = os.path.join(DATA_HOME, 'Pretrained', 'sgns.renmin.word.bz2')
    model = KeyedVectors.load_word2vec_format(path_to_model, binary=False)
    vec_china = model['中国']
    logging.info(f"中国: {vec_china}")  #


if __name__ == '__main__':
    # load_third_part_wv_en()
    load_third_part_wv_zh()
