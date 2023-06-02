"""
文件名: Code/Chapter08/C03_CLSTM/CLSTM.py
创建时间: 2023/5/31 8:29 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest

Zhou C, Sun C, Liu Z, et al. A C-LSTM neural network for text classification[J]. arXiv preprint arXiv:1511.08630, 2015.
"""
import torch.nn as nn
import torch


class CLSTM(nn.Module):
    def __init__(self, config):
        super(CLSTM, self).__init__()
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
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.conv = nn.Conv2d(1, config.out_channels,
                              kernel_size=(config.window_size, config.embedding_size))
        self.rnn = rnn_cell(config.out_channels, config.hidden_size, config.num_layers,
                            batch_first=True, bidirectional=config.bidirectional)
        self.classifier = nn.Sequential(nn.Linear(out_hidden_size, config.num_classes))

    def forward(self, x, labels=None):
        """
        :param x: [batch_size, src_len]
        :param labels: [batch_size]
        :return:
        """
        x = self.token_embedding(x)  # [batch_size, src_len, embedding_size]
        x = torch.unsqueeze(x, dim=1)  # [batch_size, 1, src_len, embedding_size]
        feature_maps = self.conv(x).squeeze(-1)  # [batch_size, out_channels, src_len-window_size+1]
        feature_maps = feature_maps.transpose(1, 2)  # [batch_size, src_len-window_size+1, out_channels]
        x, _ = self.rnn(feature_maps)  # [batch_size, src_len-window_size+1, out_hidden_size]
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
        self.vocab_size = 8
        self.embedding_size = 16
        self.out_channels = 32
        self.window_size = 3
        self.hidden_size = 128
        self.num_layers = 1
        self.cell_type = 'LSTM'
        self.bidirectional = False
        self.cat_type = 'last'


if __name__ == '__main__':
    config = ModelConfig()
    model = CLSTM(config)
    x = torch.randint(0, config.vocab_size, [2, 10], dtype=torch.long)
    label = torch.randint(0, config.num_classes, [2], dtype=torch.long)
    loss, logits = model(x, label)
    print(loss)
    print(logits)
