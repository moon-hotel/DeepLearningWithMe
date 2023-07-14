"""
文件名: Code/Chapter09/C01_Word2Vec/visualization.py
创建时间: 2023/7/13 19:43 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from gensim.models import KeyedVectors
from gensim import matutils
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import logging
import random
import sys
import os

sys.path.append('../../')
from utils import DATA_HOME


def reduce_dimensions():
    path_to_model = os.path.join(DATA_HOME, 'Pretrained', 'sgns.renmin.word.bz2')
    model = KeyedVectors.load_word2vec_format(path_to_model, binary=False, limit=2500)  # 只载入前2500个词
    num_dimensions = 2  # 降维的维度
    vectors = np.asarray(model.vectors)[1000:2500]  # 由于前面的词有部分介词虚词之类的，所以挑选后面部分
    labels = np.asarray(model.index_to_key)[1000:2500]
    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)  # 降维
    vectors = tsne.fit_transform(vectors)
    print(vectors)
    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def plot_with_matplotlib():
    x_vals, y_vals, labels = reduce_dimensions()
    random.seed(0)
    plt.figure(figsize=(8, 8))
    plt.scatter(x_vals, y_vals)
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 70)  # 由于把每个词的标签都展示出来过多，所以这里随机选择70个词的标签进行展示
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认中文字体 Windows
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
    for i in selected_indices:  # 展示标签
        plt.annotate(labels[i], (x_vals[i], y_vals[i]), fontsize=14)
    plt.tight_layout()
    plt.show()


def vector_relation():
    path_to_model = os.path.join(DATA_HOME, 'Pretrained', 'GoogleNews-vectors-negative300.bin.gz')
    model = KeyedVectors.load_word2vec_format(path_to_model, binary=True, limit=50000)  # 只载入前2500个词
    # most_sim = model.most_similar(['queen'],topn=3)
    # # [('queens', 0.739944338798523), ('princess', 0.7070532441139221), ('king', 0.6510956883430481)]
    vec_king, vec_man = model['king'], model['man']
    vec_queen, vec_woman = model['queen'], model['woman']
    result = vec_king - vec_man + vec_woman
    sim1 = np.dot(matutils.unitvec(result), matutils.unitvec(model['queen']))
    sim2 = model.similarity('queen', 'queens')
    logging.info(f"queen和推算结果的相似度为: {sim1:.4f}")  # queen和推算结果的相似度为: 0.7301
    logging.info(f"queen和queens的相似度为: {sim2:.4f}")  # queen和queens的相似度为: 0.7399
    sim = np.dot(matutils.unitvec(vec_king - vec_man), matutils.unitvec(vec_queen - vec_woman))
    logging.info(f"相似度为: {sim:.4f}")  # 相似度为: 0.7580


if __name__ == '__main__':
    # plot_with_matplotlib()
    vector_relation()
