"""
文件名: Code/Chapter08/C01_TextCNN/TextCNN.py
创建时间: 2023/5/25 7:36 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn



class TextCNN(nn.Module):
    def __init__(self, vocab_size=2000, embedding_size=512,
                 window_size=None, out_channels=2, fc_hidden_size=128, num_classes=10):
        super(TextCNN, self).__init__()
        if window_size is None:
            window_size = [2, 3, 4, 5]
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.out_channels = out_channels
        self.fc_hidden_size = fc_hidden_size
        self.num_classes = num_classes
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.convs = [nn.Conv2d(1, out_channels, kernel_size=(k, embedding_size)) for k in window_size]
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(len(self.window_size) * self.out_channels, self.fc_hidden_size),
            nn.Linear(self.fc_hidden_size, self.num_classes))

    def forward(self, x, labels=None):
        """

        :param x: [batch_size, src_len]
        :param labels: [batch_size]
        :return:
        """
        x = self.token_embedding(x)  # [batch_size, src_len, embedding_size]
        x = torch.unsqueeze(x, dim=1)  # [batch_size, 1, src_len, embedding_size]
        features = []
        for conv in self.convs:
            feature = self.max_pool(conv(x))  # [batch_size, out_channels, 1, 1]
            features.append(feature.squeeze(-1).squeeze(-1))  # [batch_size, out_channels]
        features = torch.cat(features, dim=1)  # [batch_size, out_channels*len(window_size)]
        logits = self.classifier(features)  # [batch_size, num_classes]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


if __name__ == '__main__':
    x = torch.tensor([[1, 2, 3, 2, 0, 1],
                      [2, 2, 2, 1, 3, 1]])
    labels = torch.tensor([0, 3])
    model = TextCNN(vocab_size=5, embedding_size=3, hidden_size=6)
    loss, logits = model(x, labels)
    print(loss, logits)
