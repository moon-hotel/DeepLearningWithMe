import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
from batch_normalization import BatchNormalization


class Conv(nn.Module):
    def __init__(self, ):
        super(Conv, self).__init__()
        self.conv1 = nn.Sequential(  # [n,1,28,28]
            nn.Conv2d(1, 32, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3, padding=1),
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3, padding=1),
        )
        self.conv4 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3, padding=1),
        )
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(32  * 1 * 1, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        output1 = self.conv1(img)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        output4 = self.conv4(output3)
        output5 = self.fc(output4)
        return output5, output4, output3, output2, output1,


class ConvBN(nn.Module):
    def __init__(self, ):
        super(ConvBN, self).__init__()
        self.conv1 = nn.Sequential(  # [n,1,28,28]
            nn.Conv2d(1, 32, 3, padding=1),
            BatchNormalization(32, 4),
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3, padding=1),
            BatchNormalization(32, 4),
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3, padding=1),
            BatchNormalization(32, 4),
        )
        self.conv4 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3, padding=1),
            BatchNormalization(32, 4),
        )
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(32 * 1 * 1, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        output1 = self.conv1(img)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        output4 = self.conv4(output3)
        output5 = self.fc(output4)
        return output5, output4, output3, output2, output1,


def load_dataset():
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',
                                                    train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',
                                                   train=False, download=True,
                                                   transform=transforms.ToTensor())
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
        self.use_bn = use_bn
        self.net = ConvBN()
        if not use_bn:
            self.net = Conv()

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
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)  # 定义优化器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)
        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(train_iter):
                x, y = x.to(device), y.to(device)
                logits, feature_map4, feature_map3, feature_map2, feature_map1 = self.net(x)
                if epoch == self.epochs - 1 and i == 0:
                    print(feature_map1.shape)
                    np.save(f"feature_maps1_{self.use_bn}_epoch{epoch}.npy", feature_map1.cpu().detach().numpy())
                    np.save(f"feature_maps2_{self.use_bn}_epoch{epoch}.npy", feature_map2.cpu().detach().numpy())
                    np.save(f"feature_maps3_{self.use_bn}_epoch{epoch}.npy", feature_map3.cpu().detach().numpy())
                    np.save(f"feature_maps4_{self.use_bn}_epoch{epoch}.npy", feature_map4.cpu().detach().numpy())
                l = loss(logits, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()  # 执行梯度下降
                if i % 50 == 0:
                    acc = (logits.argmax(1) == y).float().mean()
                    print("Epochs[{}/{}]---batch[{}/{}]---acc {:.4}---loss {:.4}".format(
                        epoch, self.epochs, len(mnist_train) // self.batch_size, i, acc, l.item()))
            self.net.eval()
            acc = self.evaluate(test_iter, device)
            print("Epochs[{}/{}]--acc on test {:.4}".format(epoch, self.epochs, acc))
            self.net.train()

    def evaluate(self, data_iter, device):
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            for x, y in data_iter:
                x, y = x.to(device), y.to(device)
                logits = self.net(x)[0]
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)
            return acc_sum / n




if __name__ == '__main__':

    model = MyModel(batch_size=1, learning_rate=0.1, epochs=20, use_bn=False)
    model.train()
    #
    # model = MyModel(batch_size=512, learning_rate=0.1, epochs=20, use_bn=True)
    # model.train()
