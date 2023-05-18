"""
文件名: Code/Chapter07/C07_CharRNNPoetry/CharRNN.py
创建时间: 2023/5/15 7:29 下午
"""

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, vocab_size=2000, embedding_size=64, hidden_size=128,
                 num_layers=2, cell_type='LSTM', bidirectional=True, PAD_IDX=1):
        """

        :param vocab_size: 指代的是词表的长度
        :param embedding_size: 指词向量的维度
        :param hidden_size:
        :param num_layers:
        :param cell_type: 'RNN'、'LSTM' 'GRU'
        :param bidirectional: False or True
        """
        super(CharRNN, self).__init__()
        if cell_type == 'RNN':
            rnn_cell = nn.RNN
        elif cell_type == 'LSTM':
            rnn_cell = nn.LSTM
        elif cell_type == 'GRU':
            rnn_cell = nn.GRU
        else:
            raise ValueError("Unrecognized RNN cell type: " + cell_type)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.PAD_IDX = PAD_IDX
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.rnn = rnn_cell(self.embedding_size, self.hidden_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=self.bidirectional)
        self.classifier = nn.Sequential(nn.LayerNorm(self.hidden_size),
                                        nn.Linear(self.hidden_size, self.hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.hidden_size, self.vocab_size))

    def forward(self, x, labels=None):
        """
        :param x: [batch_size, src_len]
        :param labels: [batch_size, tgt_len]
        :return: logits: [batch_size, src_len, vocab_size]
        """
        x = self.token_embedding(x)  # [batch_size, src_len, embedding_size]
        x, _ = self.rnn(x)  # [batch_size, src_len, hidden_size]
        if self.bidirectional:
            forward = x[:, :, :self.hidden_size]
            backward = x[:, :, -self.hidden_size:]
            x = 0.5 * (forward + backward)
        logits = self.classifier(x)  # [batch_size, src_len, vocab_size]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.PAD_IDX)
            loss = loss_fct(logits.reshape(-1, self.vocab_size), labels.reshape(-1))
            return loss, logits
        else:
            return logits


if __name__ == '__main__':
    input_size = 8
    model = CharRNN(input_size)
    x = torch.randint(0, input_size, [2, 3], dtype=torch.long)
    label = torch.randint(0, input_size, [2, 3], dtype=torch.long)
    loss, logits = model(x, label)
    # print(loss)
    # print(logits)

    out = model(torch.tensor([[5,2]]))
    print(out)
    print(out.shape)

