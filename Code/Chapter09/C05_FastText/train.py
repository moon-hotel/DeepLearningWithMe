"""
文件名: Code/Chapter09/C05_FastText/train.py
创建时间: 2023/7/22 11:57 上午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import sys

sys.path.append('../../')
from utils import SougoNews
from utils import logger_init
import logging
import fasttext


class ModelConfig(object):
    def __init__(self):
        self.epochs = 2
        self.model = 'cbow'  # skipgram
        self.learning_rate = 5e-3
        self.vector_size = 50
        self.window = 5  # size of the context window
        self.min_count = 10
        self.neg = 5
        self.minn = 3
        self.maxn = 6
        self.model_save_path = f'sougou_vec_{self.vector_size}.bin'
        logger_init(log_file_name='log', log_level=logging.INFO, log_dir='log')
        logging.info("### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def train_fasttext(config):
    data_loader = SougoNews()
    logging.info(f" ## 模型开始训练......")
    model = fasttext.train_unsupervised(data_loader.corpus_path, model=config.model, dim=config.vector_size,
                                        minCount=config.min_count, epoch=config.epochs, minn=config.minn,
                                        lr=config.learning_rate, neg=config.neg, maxn=config.maxn,
                                        ws=config.window)
    model.save_model(config.model_save_path)

    vec = model.get_word_vector("中国")
    logging.info(model.get_dimension())
    logging.info(model.get_subwords("跟我一起学深度学习"))
    logging.info(vec)


if __name__ == '__main__':
    config = ModelConfig()
    train_fasttext(config)
