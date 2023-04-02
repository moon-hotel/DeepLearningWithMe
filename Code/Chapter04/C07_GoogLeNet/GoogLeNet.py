"""
文件名: Code/Chapter04/C07_GoogLeNet/GoogLeNet.py
创建时间: 2023/4/2 9:23 上午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch.nn as nn
import torch


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)


class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3reduce, ch3x3, ch5x5reduce, ch5x5, pool_proj):
        """
        :param in_channels: 上一层输入的通道数
        :param ch1x1:  1x1卷积的个数
        :param ch3x3reduce:  3x3之前1x1卷积的个数
        :param ch3x3:        3x3卷积的个数
        :param ch5x5reduce:  5x5之前1x1卷积的个数
        :param ch5x5:        5x5卷积的个数
        :param pool_proj:    池化后1x1卷积的个数
        """
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(BasicConv2d(in_channels, ch3x3reduce, kernel_size=1),
                                     BasicConv2d(ch3x3reduce, ch3x3, kernel_size=3, padding=1))
        self.branch3 = nn.Sequential(BasicConv2d(in_channels, ch5x5reduce, kernel_size=1),
                                     BasicConv2d(ch5x5reduce, ch5x5, kernel_size=5, padding=2))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                     BasicConv2d(in_channels, pool_proj, kernel_size=1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3):
        super(GoogLeNet, self).__init__()
        s1 = nn.Sequential(BasicConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        s2 = nn.Sequential(BasicConv2d(64, 64, kernel_size=1, stride=1),
                           BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # in_channels, ch1x1, ch3x3reduce, ch3x3, ch5x5reduce, ch5x5, pool_proj):
        s3 = nn.Sequential(Inception(192, 64, 96, 128, 16, 32, 32),  # inception3a
                           Inception(256, 128, 128, 192, 32, 96, 64),  # inception3b
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        s4 = nn.Sequential(Inception(480, 192, 96, 208, 16, 48, 64),  # inception4a
                           Inception(512, 160, 112, 224, 24, 64, 64),  # inception4b
                           Inception(512, 128, 128, 256, 24, 64, 64),  # inception4c
                           Inception(512, 112, 144, 288, 32, 64, 64),  # inception4d
                           Inception(528, 256, 160, 320, 32, 128, 128),  # inception4e
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        s5 = nn.Sequential(Inception(832, 256, 160, 320, 32, 128, 128),  # inception5a
                           Inception(832, 384, 192, 384, 48, 128, 128),  # inception5b
                           nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                           nn.Dropout(0.5), nn.Linear(1024, num_classes))
        self.google_net = nn.Sequential(s1, s2, s3, s4, s5)

    def forward(self, x, labels=None):
        logits = self.google_net(x)
        # N x 1000 (num_classes)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


if __name__ == '__main__':
    model = GoogLeNet(10,3)
    print(model)
    x = torch.randn(2, 3, 224, 224)
    for seq in model.children():
        for layer in seq:
            x = layer(x)
            print(f"网络层: {layer.__class__.__name__}, 输出形状: {x.shape}")
