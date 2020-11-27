import torch.nn as nn
import torch


class PrintLayer(nn.Module):
    def __init__(self, name):
        super(PrintLayer, self).__init__()
        self.name = name

    def forward(self, x):
        print(self.name, x.shape)
        return x


class NiN(nn.Module):
    def __init__(self,init_weights=True):
        super(NiN, self).__init__()

        self.nin = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=160, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=160, out_channels=96, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),


            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2 ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=10, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8),
            nn.Flatten()
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        return self.nin(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    X = torch.rand(1, 3, 32, 32)
    model = NiN()
    # print(model)
    y = model(X)
    # print(y)
