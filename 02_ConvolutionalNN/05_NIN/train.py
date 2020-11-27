from nin import NiN
import torchvision
import torch
import torch.nn as nn


def load_dataset(batch_size, resize=None):
    trans = []
    if resize:  # 将输入的28*28的图片，resize成224*224的形状
        trans.append(torchvision.transforms.Resize(size=resize, interpolation=1))
    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.CIFAR10(root='~/Datasets/CIFAR10',
                                                    train=True, download=True,
                                                    transform=transform)
    mnist_test = torchvision.datasets.CIFAR10(root='~/Datasets/CIFAR10',
                                                   train=False, download=True,
                                                   transform=transform)
    train_iter = torch.utils.data.DataLoader(mnist_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=1)
    test_iter = torch.utils.data.DataLoader(mnist_test,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=1)
    return train_iter, test_iter


class MyModel:
    def __init__(self,
                 model=None,
                 batch_size=128,
                 epochs=500,
                 learning_rate=0.001):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.net = model

    def train(self):
        train_iter, test_iter = load_dataset(batch_size=self.batch_size,resize=None)
        loss = nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)  # 定义优化器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)
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
                    print("Epochs[{}/{}]---batch {}---acc {:.4}---loss {:.4}".format(
                        epoch + 1, self.epochs, i, acc, l))
            self.net.eval()  # 切换到评估模式
            print("Epochs[{}/{}]--acc on test {:.4}".format(epoch + 1, self.epochs,
                                                            self.evaluate(test_iter, self.net, device)))
            self.net.train()  # 切回到训练模式
            if (epoch+1)%100==0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.8
                    print("learning rate:", param_group['lr'])

    @staticmethod
    def evaluate(data_iter, net, device):
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            for x, y in data_iter:
                x, y = x.to(device), y.to(device)
                logits = net(x)
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)
            return acc_sum / n


if __name__ == '__main__':
    model = NiN()
    nin = MyModel(model=model, batch_size=128, epochs=800, learning_rate=0.0004)
    nin.train()
    # print(model)
