import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from LeNet5 import LeNet5BN
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
                 batch_size=128,
                 epochs=3,
                 learning_rate=0.001,
                 use_bn=True):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.net = LeNet5BN()
        if not use_bn:
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
        loss = nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)  # 定义优化器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)
        loss_his = []
        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(train_iter):
                x, y = x.to(device), y.to(device)
                logits = self.net(x)[0]
                l = loss(logits, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()  # 执行梯度下降
                loss_his.append(round(l.item(), 2))
                if i % 50 == 0:
                    acc = (logits.argmax(1) == y).float().mean()
                    print("Epochs[{}/{}]---batch[{}/{}]---acc {:.4}---loss {:.4}".format(
                        epoch, self.epochs, len(mnist_train) // self.batch_size, i, acc, l.item()))
            self.net.eval()
            acc = self.evaluate(test_iter, device)
            print("Epochs[{}/{}]--acc on test {:.4}".format(epoch, self.epochs, acc))
            self.net.train()
        return loss_his

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
    model = MyModel(batch_size=512, learning_rate=0.01, epochs=20)
    history_loss_bn = model.train()
    model = MyModel(batch_size=512, learning_rate=0.01, epochs=20, use_bn=False)
    history_loss = model.train()

    print("history_loss_bn = ",history_loss_bn)
    print("history_loss = ",history_loss)
