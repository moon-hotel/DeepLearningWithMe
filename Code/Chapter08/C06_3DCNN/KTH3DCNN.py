"""
文件名: Code/Chapter08/C06_3DCNN/KTH3DCNN.py
创建时间: 2023/6/18 4:47 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch.nn as nn
import torch


class KTH3DCNN(nn.Module):
    def __init__(self, config):
        super(KTH3DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(config.in_channels, out_channels=32, kernel_size=3, stride=1, padding=(0, 1, 1)),  # 帧数减2
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, out_channels=64, kernel_size=3, stride=1, padding=(0, 1, 1)),  # 帧数减2
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=(0, 1, 1)),  # 长宽减半，帧数减2
            nn.Conv3d(64, out_channels=128, kernel_size=3, stride=1, padding=(0, 1, 1)),  # 帧数减2
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)))  # 帧数、长宽均为1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=config.num_classes))

    def forward(self, x, labels=None):
        """
        :param x: [batch_size, in_channels, frames, height, width]
        :param labels:
        :return:
        """
        x = self.features(x)
        logits = self.classifier(x)  # [batch_size, num_classes]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class ModelConfig(object):
    def __init__(self):
        self.num_classes = 6
        self.in_channels = 3
        self.batch_size = 8
        self.height = 60
        self.width = 80
        self.frames = 15


if __name__ == '__main__':
    config = ModelConfig()
    x = torch.randn([config.batch_size, config.in_channels, config.frames,
                     config.height, config.width])
    label = torch.randint(0, config.num_classes, [config.batch_size], dtype=torch.long)
    model = KTH3DCNN(config)
    loss, logits = model(x, label)
    print(logits)
    print(f"输入形状：{x.shape}")
    for seq in model.children():
        for layer in seq:
            x = layer(x)
            print(f"网络层: {layer.__class__.__name__}, 输出形状: {x.shape}")
