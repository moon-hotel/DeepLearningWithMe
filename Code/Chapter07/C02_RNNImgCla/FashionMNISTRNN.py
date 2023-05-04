"""
文件名: Code/Chapter07/C01_RNN/FashionMNISTRNN.py
创建时间: 2023/4/27 8:08 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn
import sys

sys.path.append('../../')
from Chapter06.C04_LN.layer_normalization import LayerNormalization


class FashionMNISTRNN(nn.Module):
    def __init__(self, input_size=28, hidden_size=128,
                 num_layers=1, num_classes=10):
        super(FashionMNISTRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.layer_norm = LayerNormalization(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, labels=None):
        x = x.squeeze(1)  # [batch_size,1,28,28] --> [batch_size,28,28]
        x, _ = self.rnn(x)  # input: [batch_size, time_steps, input_size]
        x = self.layer_norm(x)  # x: [batch_size, time_steps, hidden_size]
        logits = self.fc(x[:, -1].squeeze(1))
        # 取最后一个时刻进行分类，[batch_size, 1,hidden_size]-->[batch_size,hidden_size]
        # logits: [batch_size, hidden_size]
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
