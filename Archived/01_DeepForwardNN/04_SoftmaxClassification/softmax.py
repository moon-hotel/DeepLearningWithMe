import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np


def loadDataset():
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                                   transform=transforms.ToTensor())
    return mnist_train, mnist_test


def softmax(x):
    s = torch.exp(x)
    return s / torch.sum(s, dim=1, keepdim=True)


def crossEntropy(y_true, logits):
    c = -torch.log(logits.gather(1, y_true.reshape(-1, 1)))
    return torch.sum(c)


def accuracy(y_true, logits):
    acc = (logits.argmax(1) == y_true).float().mean()
    return acc.item()


def forward(x, input_nodes, w, b):
    y = torch.matmul(x.reshape(-1, input_nodes), w) + b
    return softmax(y)


def evaluate(data_iter, forward, input_nodes, w, b):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        logits = forward(x, input_nodes, w, b)
        acc_sum += (logits.argmax(1) == y).float().sum().item()
        n += len(y)
    return acc_sum / n


def gradientDescent(params, lr):
    for param in params:
        param.data -= lr * param.grad
        param.grad.zero_()


def train(mnist_train, mnist_test):
    input_nodes = 28 * 28
    output_nodes = 10
    epochs = 8000
    lr = 0.002
    batch_size = 128
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=2)

    w = torch.tensor(np.random.normal(0, 0.5, [input_nodes, output_nodes]),
                     dtype=torch.float32, requires_grad=True)
    b = torch.tensor(np.random.randn(output_nodes), dtype=torch.float32, requires_grad=True)
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_iter):
            logits = forward(x, input_nodes, w, b)
            l = crossEntropy(y, logits)
            l.backward()
            gradientDescent([w, b], lr)
            acc = accuracy(y, logits)
            if i % 50 == 0:
                print("Epochs[{}/{}]---batch[{}/{}]---acc {:.4}---loss {:.4}".format(
                    epochs, epoch, len(mnist_train) // batch_size, i, acc,l/len(y)))
        acc = evaluate(test_iter, forward, input_nodes, w, b)
        print("Epochs[{}/{}]--acc on test {:.4}".format(epochs, epoch, acc))


if __name__ == '__main__':
    mnist_train, mnist_test = loadDataset()
    train(mnist_train, mnist_test)

