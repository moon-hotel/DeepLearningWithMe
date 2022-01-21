import torch.nn as nn
import torch
import os


class LeNet5(nn.Module):
    def __init__(self, ):
        super(LeNet5, self).__init__()
        self.conv = nn.Sequential(  # [n,1,28,28]
            nn.Conv2d(1, 6, 5, padding=2),  # in_channels, out_channels, kernel_size
            nn.ReLU(),  # [n,6,24,24]
            nn.MaxPool2d(2, 2),  # kernel_size, stride  [n,6,14,14]
            nn.Conv2d(6, 16, 5),  # [n,16,10,10]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # [n,16,5,5]
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, img):
        output = self.conv(img)
        output = self.fc(output)
        return output
