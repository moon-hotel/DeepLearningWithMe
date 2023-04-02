import torch.nn as nn
import torch
from batch_normalization import BatchNormalization
import torchvision


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),

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

            nn.Linear(in_features=4096, out_features=100),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature)
        return output


class AlexNetBN(nn.Module):
    def __init__(self):
        super(AlexNetBN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            BatchNormalization(96, 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # kernel_size, stride
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            BatchNormalization(256, 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            BatchNormalization(384, 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            BatchNormalization(384, 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            BatchNormalization(256, 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 这里全连接层的输出个数比LeNet中的大数倍。使用了丢弃层Dropout来缓解过拟合
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256 * 5 * 5, out_features=4096),
            BatchNormalization(4096, 2),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            BatchNormalization(4096, 2),
            nn.ReLU(),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(in_features=4096, out_features=100),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature)
        return output





def load_dataset():
    trans = [torchvision.transforms.Resize(size=224, interpolation=1), torchvision.transforms.ToTensor()]
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.CIFAR100(root='~/Datasets/CIFAR100',
                                                train=True, download=True,
                                                transform=transform)
    mnist_test = torchvision.datasets.CIFAR100(root='~/Datasets/CIFAR100',
                                               train=False, download=True,
                                               transform=transform)
    return mnist_train, mnist_test


class MyModel:
    def __init__(self,
                 batch_size=128,
                 epochs=3,
                 learning_rate=0.001,
                 use_bn=True):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.net = AlexNet()
        if not use_bn:
            self.net = AlexNetBN()

    def train(self):
        mnist_train, mnist_test = load_dataset()
        train_iter = torch.utils.data.DataLoader(mnist_train,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 num_workers=1)
        test_iter = torch.utils.data.DataLoader(mnist_test,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=1)
        loss = nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)  # 定义优化器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)
        acc_train, acc_test = [], []
        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(train_iter):
                x, y = x.to(device), y.to(device)
                logits = self.net(x)
                l = loss(logits, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()  # 执行梯度下降
                if i % 50 == 0:
                    acc = (logits.argmax(1) == y).float().mean()
                    print("Epochs[{}/{}]---batch[{}/{}]---acc {:.4}---loss {:.4}".format(
                        epoch, self.epochs, len(mnist_train) // self.batch_size, i, acc, l.item()))
            if epoch % 1 == 0:
                self.net.eval()
                acc = self.evaluate(train_iter, device)
                acc_train.append(acc)

                acc = self.evaluate(test_iter, device)
                acc_test.append(acc)
                print("Epochs[{}/{}]--acc on test {:.4}".format(epoch, self.epochs, acc))
                self.net.train()
        return acc_train, acc_test

    def evaluate(self, data_iter, device):
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            for x, y in data_iter:
                x, y = x.to(device), y.to(device)
                logits = self.net(x)
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)
            return acc_sum / n


if __name__ == '__main__':
    model = MyModel(batch_size=256, learning_rate=0.001, epochs=30,use_bn=False)
    acc_train, acc_test = model.train()
    print("acc_train = ",acc_train)
    print("acc_test = ",acc_test)

    model = MyModel(batch_size=256, learning_rate=0.001, epochs=30, use_bn=True)
    acc_train, acc_test = model.train()
    print("acc_train_bn = ",acc_train)
    print("acc_test_bn = ",acc_test)
