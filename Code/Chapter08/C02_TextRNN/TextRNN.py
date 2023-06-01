"""
文件名: Code/Chapter08/C02_TextRNN/TextRNN.py
创建时间: 2023/5/28 8:29 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn


class TextRNN(nn.Module):
    def __init__(self, config):
        """
        :param num_classes: 分类数
        :param vocab_size: 指代的是词表的长度
        :param embedding_size: 指词向量的维度
        :param hidden_size:
        :param num_layers:
        :param cell_type: 'RNN'、'LSTM' 'GRU'
        :param bidirectional: False or True
        :param cat_type: 特征组合方式
        """
        super(TextRNN, self).__init__()
        if config.cell_type == 'RNN':
            rnn_cell = nn.RNN
        elif config.cell_type == 'LSTM':
            rnn_cell = nn.LSTM
        elif config.cell_type == 'GRU':
            rnn_cell = nn.GRU
        else:
            raise ValueError("Unrecognized RNN cell type: " + config.cell_type)
        out_hidden_size = config.hidden_size * (int(config.bidirectional) + 1)
        self.config = config
        self.token_embedding = nn.Embedding(config.top_k, config.embedding_size)
        self.rnn = rnn_cell(config.embedding_size, config.hidden_size, num_layers=config.num_layers,
                            batch_first=True, bidirectional=config.bidirectional)
        self.classifier = nn.Sequential(nn.LayerNorm(out_hidden_size),
                                        nn.Linear(out_hidden_size, out_hidden_size),
                                        nn.ReLU(inplace=True), nn.Dropout(0.5),
                                        nn.Linear(out_hidden_size, config.num_classes))

    def forward(self, x, labels=None):
        """
        :param x: [batch_size, src_len]
        :param labels: [batch_size, tgt_len]
        :return: logits: [batch_size, src_len, vocab_size]
        """
        x = self.token_embedding(x)  # [batch_size, src_len, embedding_size]
        x, _ = self.rnn(x)  # [batch_size, src_len, out_hidden_size]

        if self.config.cat_type == 'last':
            x = x[:, -1]  # [batch_size, out_hidden_size]
        elif self.config.cat_type == 'mean':
            x = torch.mean(x, dim=1)  # [batch_size, out_hidden_size]
        elif self.config.cat_type == 'sum':
            x = torch.sum(x, dim=1)  # [batch_size, out_hidden_size]
        else:
            raise ValueError("Unrecognized cat_type: " + self.cat_type)
        logits = self.classifier(x)  # [batch_size, num_classes]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class ModelConfig(object):
    def __init__(self):
        self.num_classes = 15
        self.top_k = 8
        self.embedding_size = 16
        self.hidden_size = 512
        self.num_layers = 2
        self.cell_type = 'LSTM'
        self.bidirectional = False
        self.cat_type = 'last'


if __name__ == '__main__':
    config = ModelConfig()
    model = TextRNN(config)
    x = torch.randint(0, config.top_k, [2, 3], dtype=torch.long)
    label = torch.randint(0, config.num_classes, [2], dtype=torch.long)
    loss, logits = model(x, label)
    print(loss)
    print(logits)

    # inference
    out = model(torch.tensor([[5, 2, 6, 7, 7]]))
    print(out)
    print(out.shape)
