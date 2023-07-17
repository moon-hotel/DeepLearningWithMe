from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import logging
import sys
import os

sys.path.append('../../')
from utils import DATA_HOME


def load_glove_6b50():
    glove_path = os.path.join(DATA_HOME, 'Pretrained', 'glove6b', 'glove.6B.50d.txt')
    glove_word2vec_path = os.path.join(DATA_HOME, 'Pretrained', 'glove6b',
                                       'glove.6B.50d.word2vec.txt')
    if not os.path.exists(glove_word2vec_path):
        glove2word2vec(glove_path, glove_word2vec_path)
    model = KeyedVectors.load_word2vec_format(glove_word2vec_path)
    logging.info(f"china: {model['china']}")
    sim_words = model.most_similar(['china'], topn=5)
    # china: [-0.22427   0.27427   0.054742  1.4692    0.061821 -0.51894 ......
    logging.info(f"与china最相似的前5个词为:{sim_words}")
    # [('taiwan', 0.936076283454895), ('chinese', 0.8957242369651794), ('beijing', 0.8920878171920776),
    # ('mainland', 0.8644797205924988), ('japan', 0.8428842425346375)]


if __name__ == '__main__':
    load_glove_6b50()
