import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch


def loadDataset():
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',
                                                    train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',
                                                   train=False, download=True,
                                                   transform=transforms.ToTensor())
    return mnist_train, mnist_test


def accuracy(y_true, logits):
    acc = (logits.argmax(1) == y_true).float().mean()
    return acc.item()


def evaluate(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        logits = net(x)
        acc_sum += (logits.argmax(1) == y).float().sum().item()
        n += len(y)
    return acc_sum / n


def train(mnist_train, mnist_test):
    input_nodes = 28 * 28
    hidden_nodes = 1024
    output_nodes = 10
    epochs = 5
    lr = 0.001
    batch_size = 256
    train_iter = torch.utils.data.DataLoader(mnist_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=2)
    test_iter = torch.utils.data.DataLoader(mnist_test,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=2)
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(input_nodes, hidden_nodes),
                        nn.ReLU(),
                        nn.Linear(hidden_nodes, hidden_nodes),
                        nn.ReLU(),
                        nn.Linear(hidden_nodes, output_nodes)
                        )
    loss = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 定义优化器
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_iter):
            logits = net(x)
            l = loss(logits, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()  #执行梯度下降
            if i % 50 == 0:
                acc = accuracy(y, logits)
                print("Epochs[{}/{}]---batch[{}/{}]---acc {:.4}---loss {:.4}".format(
                    epochs, epoch, len(mnist_train) // batch_size, i, acc, l))
        acc = evaluate(test_iter, net)
        print("Epochs[{}/{}]--acc on test {:.4}".format(epochs, epoch, acc))


if __name__ == '__main__':
    mnist_train, mnist_test = loadDataset()
    train(mnist_train, mnist_test)
