"""
文件名: Code/Chapter06/C03_BN/LeNet5.py
创建时间: 2023/4/7 4:26 下午
"""
import torch.nn as nn
from batch_normalization import BatchNormalization


class LeNet5(nn.Module):
    def __init__(self, ):
        super(LeNet5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            BatchNormalization(num_features=6, num_dims=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            BatchNormalization(num_features=16, num_dims=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))  # [n,16,5,5]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            BatchNormalization(num_features=120, num_dims=2),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            BatchNormalization(num_features=84, num_dims=2),
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
