import torch.nn as nn
import torch
from batch_normalization import BatchNormalization


class LeNet5BN(nn.Module):
    def __init__(self, ):
        super(LeNet5BN, self).__init__()
        self.conv = nn.Sequential(  # [n,1,28,28]
            nn.Conv2d(1, 6, 5, padding=2),  # in_channels, out_channels, kernel_size
            BatchNormalization(6, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            BatchNormalization(16, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            BatchNormalization(120, 2),
            nn.ReLU(),
            nn.Linear(120, 84),
            BatchNormalization(84, 2),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        output1 = self.conv(img)
        output2 = self.fc(output1)
        return output2, output1


class LeNet5(nn.Module):
    def __init__(self, ):
        super(LeNet5, self).__init__()
        self.conv = nn.Sequential(  # [n,1,28,28]
            nn.Conv2d(1, 6, 5, padding=2),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        output1 = self.conv(img)
        output2 = self.fc(output1)
        return output2, output1
