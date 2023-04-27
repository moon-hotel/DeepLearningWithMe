"""
文件名: Code/Chapter07/C01_RNN/FashionMNISTRNN.py
创建时间: 2023/4/27 8:08 下午
"""

import torch
import torch.nn as nn
import sys

sys.path.append('../')


class FashionMNISTRNN(nn.Module):
    def __init__(self, input_size=28, hidden_size=128,
                 num_layers=2, num_classes=10):
        super(FashionMNISTRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, labels=None):
        x = x.view(-1, 28, 28)
        x, _ = self.rnn(x)
        x = self.layer_norm(x)
        logits = self.fc(x)[:,-1]
        logits = logits.squeeze(1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


if __name__ == '__main__':
    model = FashionMNISTRNN()
    x = torch.rand([32, 1, 28, 28])
    y = model(x)
    print(y.shape)
