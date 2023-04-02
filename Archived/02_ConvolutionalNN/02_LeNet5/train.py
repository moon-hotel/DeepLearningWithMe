import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from LeNet5 import LeNet5


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
                 batch_size=64,
                 epochs=5,
                 learning_rate=0.001):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.net = LeNet5()

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
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)  # 定义优化器
        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(train_iter):
                loss, logits = self.net(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # 执行梯度下降
                if i % 50 == 0:
                    acc = (logits.argmax(1) == y).float().mean()
                    print("Epochs[{}/{}]---batch[{}/{}]---acc {:.4}---loss {:.4}".format(
                        epoch + 1, self.epochs, i, len(train_iter), acc, loss.item()))

            print("Epochs[{}/{}]--acc on test {:.4}".format(epoch, self.epochs, self.evaluate(test_iter, self.net)))

    @staticmethod
    def evaluate(data_iter, net):
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            for x, y in data_iter:
                logits = net(x)
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)
            return acc_sum / n


if __name__ == '__main__':
    model = MyModel()
    model.train()
