import torch.nn as nn
import torch


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
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

        conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3reduce, kernel_size=1),
            conv_block(ch3x3reduce, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5reduce, kernel_size=1),
            conv_block(ch5x5reduce, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()
        conv_block, inception_block = BasicConv2d, Inception

        self.conv1 = conv_block(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3 = conv_block(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # in_channels, ch1x1, ch3x3reduce, ch3x3, ch5x5reduce, ch5x5, pool_proj):
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x, labels=None):
        x = self.conv1(x)
        x = self.maxpool1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        # print(x.shape)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        # print(x.shape)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        # print(x.shape)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        # print(x.shape)
        x = torch.flatten(x, 1)
        # N x 1024
        # print(x.shape)
        x = self.dropout(x)
        logits = self.fc(x)
        # N x 1000 (num_classes)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


if __name__ == '__main__':
    m = GoogLeNet(10)
    input = torch.randn(1, 1, 224, 224)
    output = m(input)
    print(output.shape)
