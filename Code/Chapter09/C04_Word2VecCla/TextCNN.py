"""
文件名: Code/Chapter09/C04_Word2VecCla/TextCNN.py
创建时间: 2023/7/17 20:54 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest

Yoon Kim. 2014. Convolutional Neural Networks for Sentence Classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1746–1751, Doha, Qatar. Association for Computational Linguistics.
Zhang Y, Wallace B. A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification[J]. arXiv preprint arXiv:1510.03820, 2015.
"""

import torch
import torch.nn as nn
from glove_embedding import get_glove_embedding
import sys
import logging

sys.path.append('../../')
from utils import MR


class TextCNN(nn.Module):
    def __init__(self, vocab_size=2000, embedding_size=50, window_size=None,
                 out_channels=2, num_classes=10, vocab=None):
        super(TextCNN, self).__init__()
        if window_size is None:
            window_size = [3, 4, 5]
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.random_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.glove_embedding = get_glove_embedding(vocab, self.embedding_size)
        self.convs = nn.ModuleList([nn.Conv2d(2, out_channels, kernel_size=(k, embedding_size)) for k in window_size])
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(len(self.window_size) * self.out_channels, self.num_classes))

    def forward(self, x, labels=None):
        """
        :param x: [batch_size, src_len]
        :param labels: [batch_size]
        :return:
        """
        x_random = self.random_embedding(x)  # [batch_size,src_len, embedding_size]
        x_random = torch.unsqueeze(x_random, dim=1)  # [batch_size, 1, src_len, embedding_size]
        x_glove = self.glove_embedding(x)
        x_glove = torch.unsqueeze(x_glove, dim=1)  # [batch_size, 1, src_len, embedding_size]
        embedded_x = torch.cat([x_random, x_glove], dim=1)  # [batch_size,1, src_len, embedding_size]
        features = []
        for conv in self.convs:
            feature = self.max_pool(conv(embedded_x))  # [batch_size, out_channels, 1, 1]
            features.append(feature.squeeze(-1).squeeze(-1))  # [batch_size, out_channels]
        features = torch.cat(features, dim=1)  # [batch_size, out_channels*len(window_size)]
        logits = self.classifier(features)  # [batch_size, num_classes]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class ModelConfig(object):
    def __init__(self):
        self.batch_size = 128
        self.epochs = 50
        self.num_classes = 2
        self.top_k = 3000
        self.window_size = [3, 4, 5]
        self.embedding_size = 50
        self.out_channels = 3
        self.cut_words = False

        # 判断是否存在GPU设备，其中0表示指定第0块设备
        logging.info("### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


if __name__ == '__main__':
    config = ModelConfig()
    x = torch.tensor([[1, 2, 3, 2, 0, 1],
                      [2, 2, 2, 1, 3, 1]], dtype=torch.long)

    labels = torch.tensor([0, 1])
    dataloader = MR(top_k=config.top_k,
                    max_sen_len=None,
                    batch_size=config.batch_size, cut_words=config.cut_words)
    vocab = dataloader.get_vocab()
    model = TextCNN(vocab_size=config.top_k, embedding_size=config.embedding_size,
                    window_size=config.window_size, out_channels=config.out_channels,
                    num_classes=config.num_classes, vocab=vocab)
    loss, logits = model(x, labels)
    print(loss, logits)
