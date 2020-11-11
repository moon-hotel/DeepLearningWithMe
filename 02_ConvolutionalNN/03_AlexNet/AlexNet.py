import torch.nn as nn
import torch


class PrintLayer(nn.Module):
    def __init__(self, name):
        super(PrintLayer, self).__init__()
        self.name = name

    def forward(self, x):
        print(self.name, x.shape)
        return x


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),
            # PrintLayer(name="①卷积层："),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # kernel_size, stride
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 这里全连接层的输出个数比LeNet中的大数倍。使用了丢弃层Dropout来缓解过拟合
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256 * 5 * 5, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(in_features=4096, out_features=10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature)
        return output


if __name__ == '__main__':
    model = AlexNet()
    x = torch.rand(32, 1, 224, 224)
    model(x)
    print(model)

