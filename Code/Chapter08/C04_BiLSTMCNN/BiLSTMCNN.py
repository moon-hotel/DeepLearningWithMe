"""
文件名: Code/Chapter08/C04_BiLSTMCNN/BiLSTMCNN.py
创建时间: 2023/6/3 2:35 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest

Lin S, Xie H, Yu L C, et al. SentiNLP at IJCNLP-2017 Task 4: Customer feedback analysis using a Bi-LSTM-CNN model[C]
"""
import torch
import torch.nn as nn


class BiLSTMCNN(torch.nn.Module):
    def __init__(self, config=None):
        super(BiLSTMCNN, self).__init__()
        if config.cell_type == 'RNN':
            rnn_cell = nn.RNN
        elif config.cell_type == 'LSTM':
            rnn_cell = nn.LSTM
        elif config.cell_type == 'GRU':
            rnn_cell = nn.GRU
        else:
            raise ValueError("Unrecognized RNN cell type: " + config.cell_type)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn = rnn_cell(config.embedding_size, config.hidden_size, config.num_layers,
                            batch_first=True, bidirectional=True)
        self.cnn = nn.Conv2d(2, config.out_channels, [config.kernel_size, config.hidden_size])
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(config.out_channels, config.num_classes))

    def forward(self, x, labels=None):
        """
        :param x: [batch_size, src_len]
        :param labels: [batch_size, tgt_len]
        :return: logits: [batch_size, src_len, vocab_size]
        """
        x = self.token_embedding(x)  # [batch_size, src_len, embedding_size]
        x, _ = self.rnn(x)  # [batch_size, src_len, 2 * hidden_size]
        x = torch.reshape(x, (x.shape[0], x.shape[1], 2, -1))  # [batch_size, src_len, 2 , hidden_size]
        x = x.transpose(1, 2)  # [batch_size, 2, src_len, hidden_size]
        x = self.cnn(x)  # [batch_size, out_channels, src_len - kernel_size + 1, 1]
        x = self.max_pool(x)  # [batch_size, out_channels, 1, 1]
        x = torch.flatten(x, start_dim=1)  # [batch_size, out_channels]
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
        self.vocab_size = 8
        self.embedding_size = 16
        self.hidden_size = 512
        self.num_layers = 2
        self.cell_type = 'LSTM'
        self.cat_type = 'last'
        self.kernel_size = 3
        self.out_channels = 64


if __name__ == '__main__':
    config = ModelConfig()
    model = BiLSTMCNN(config)

    x = torch.randint(0, config.vocab_size, [2, 3], dtype=torch.long)
    label = torch.randint(0, config.num_classes, [2], dtype=torch.long)
    loss, logits = model(x, label)
    print(loss)
    print(logits)
