"""
文件名: Code/Chapter04/C08_ResNet/ResNet.py
创建时间: 2023/4/11 8:35 下午
"""
import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    """
    输入: x
    x -> [conv3 + bn + relu + conv + bn + x + relu]
    """

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()
        self.downsample = downsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels=3, block=None, layers=None, num_classes=1000):
        super().__init__()
        # input shape [1,3,224,224]
        self.last_layer_channels = 64
        self.layer0 = nn.Sequential(nn.Conv2d(in_channels, self.last_layer_channels,
                                              kernel_size=7, stride=2, padding=3, bias=False),
                                    # torch.Size([1, 64, 112, 112])
                                    nn.BatchNorm2d(self.last_layer_channels),
                                    # torch.Size([1, 64, 112, 112])
                                    nn.ReLU(inplace=True),
                                    # torch.Size([1, 64, 112, 112])
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # torch.Size([1, 64, 56, 56])

        self.layer1 = self._make_layer(block, 64, layers[0])  # torch.Size([1, 64, 56, 56])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # torch.Size([1, 128, 28, 28])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # torch.Size([1, 256, 14, 14])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # torch.Size([1, 512, 7, 7])
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),  # torch.Size([1, 512, 1, 1])
                                        nn.Flatten(),  # torch.Size([1, 512])
                                        nn.Linear(512, num_classes))  # torch.Size([1, num_classes])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1:  # stride = 2
            downsample = nn.Sequential(
                nn.Conv2d(self.last_layer_channels, channels, 1, stride),
                nn.BatchNorm2d(channels))
        layers.append(block(self.last_layer_channels, channels, downsample, stride))
        self.last_layer_channels = channels
        for _ in range(1, blocks):
            layers.append(block(self.last_layer_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x, labels=None):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        logits = self.classifier(x)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


def resnet18(num_classes=1000, in_channels=3):
    model = ResNet(in_channels, BasicBlock, [2, 2, 2, 2],
                   num_classes=num_classes)
    return model


if __name__ == '__main__':
    model = resnet18()
    print(model)
    x = torch.rand(1, 3, 224, 224)
    for seq in model.children():
        for layer in seq:
            x = layer(x)
            print(f"网络层: {layer.__class__.__name__}, 输出形状: {x.shape}")
