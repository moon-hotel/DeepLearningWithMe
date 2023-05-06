"""
文件名: Code/Chapter07/C03_RNNNewsCla/NewsRNN.py
创建时间: 2023/5/6 10:37 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn
import sys


class NewsRNN(nn.Module):
    def __init__(self, input_size=28, hidden_size=128,
                 num_layers=2, num_classes=10):
        super(NewsRNN, self).__init__()
        self.input_size = input_size
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity='relu',
                          num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_size),
                                        nn.Linear(hidden_size, hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_size, num_classes))

    def forward(self, x, labels=None):
        x = torch.nn.functional.one_hot(x, self.input_size).type(torch.float32)
        x, _ = self.rnn(x)  # input: [batch_size, time_steps, input_size]
        # x: [batch_size, time_steps, hidden_size]
        logits = self.classifier(x[:, -1].squeeze(1))
        # 取最后一个时刻进行分类，[batch_size, 1,hidden_size]---squeeze-->[batch_size,hidden_size]
        # logits: [batch_size, hidden_size]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


if __name__ == '__main__':
    input_size = 8
    model = NewsRNN(input_size)
    x = torch.randint(0, input_size, [2, 3], dtype=torch.long)
    y = model(x)
    print(y.shape)
