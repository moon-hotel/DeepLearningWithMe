"""
文件名: Code/Chapter09/C04_Word2VecCla/glove_embedding.py
创建时间: 2023/7/17 22:11 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn
import logging
import sys
import os
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np

sys.path.append('../../')
from utils import DATA_HOME
from utils import MR


def get_glove_embedding(vocab=None, embedding_size=50):
    if embedding_size not in [50, 100, 200, 300]:
        raise ValueError(f"embedding_size must in  [50,100,200,300], but got {embedding_size}")
    glove_path = os.path.join(DATA_HOME, 'Pretrained', 'glove6b', f'glove.6B.{embedding_size}d.txt')
    glove_word2vec_path = os.path.join(DATA_HOME, 'Pretrained', 'glove6b',
                                       f'glove.6B.{embedding_size}d.word2vec.txt')
    if not os.path.exists(glove_word2vec_path):
        glove2word2vec(glove_path, glove_word2vec_path)
    model = KeyedVectors.load_word2vec_format(glove_word2vec_path)
    assert embedding_size == model.vector_size, "embedding_size和model.vector_size不一致"
    vocab_size = len(vocab)
    embedding_weight = []
    num_not_found = 0
    for word, _ in vocab.items():
        if word in model:
            embedding_weight.append(model[word])
        else:
            num_not_found += 1
            embedding_weight.append(np.random.uniform(-1, 1, embedding_size))
    logging.info(f" ## 从GloVe构建词嵌入层完毕，词表一共有{len(vocab)}个词，"
                 f"其中有{num_not_found}个未找到，随机初始化！")
    embedding_weight = np.array(embedding_weight)  #
    embedding_weight = torch.tensor(embedding_weight, dtype=torch.float32)
    # logging.info(model.vectors[0])
    return nn.Embedding(vocab_size, embedding_size, _weight=embedding_weight)


if __name__ == '__main__':
    dataloader = MR(top_k=2000, max_sen_len=None, batch_size=4, is_sample_shuffle=True, cut_words=False)
    train_iter, val_iter = dataloader.load_train_val_test_data(is_train=True)
    vocab = dataloader.get_vocab()
    embedding = get_glove_embedding(vocab, 150)
    input = torch.LongTensor([[0, 2, 0, 5]])
    print(embedding(input))
