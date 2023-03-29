"""
文件名: Code/Chapter04/C03_LeNet5/LeNet5.py
创建时间: 2023/2/26 9:29 上午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch.nn as nn
import torch


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class LeNet5(nn.Module):
    def __init__(self, ):
        super(LeNet5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),  # [n,6,24,24]
            nn.MaxPool2d(2, 2),  # kernel_size, stride  [n,6,14,14]
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # [n,16,10,10]
            # PrintLayer(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # [n,16,5,5]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10))

    def forward(self, img, labels=None):
        output = self.conv(img)
        logits = self.fc(output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


if __name__ == '__main__':
    model = LeNet5()
    print(model)
    print(model.conv[3])
    x = torch.rand(32, 1, 28, 28)
    logits = model(x)
    print(f"模型输出结果的形状为：{logits.shape}\n")
    no_layer = 0
    for seq in model.children():
        for layer in seq:
            x = layer(x)
            if hasattr(layer, 'weight'):
                no_layer += 1
                print(f"网络层{no_layer}: {layer.__class__.__name__}, 输出形状: {x.shape}")
            else:
                print(f"网络层: {layer.__class__.__name__}, 输出形状: {x.shape}")
