import torchvision
import torchvision.transforms as transforms
from transformers import get_cosine_schedule_with_warmup
import torch
from LeNet5 import LeNet5
import matplotlib.pyplot as plt
import os


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
                 epochs=3,
                 learning_rate=0.01):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_save_path = 'model.pt'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = LeNet5()

    def train(self):
        self.model = self.model.to(self.device)
        mnist_train, mnist_test = load_dataset()
        train_iter = torch.utils.data.DataLoader(mnist_train,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 num_workers=1)
        test_iter = torch.utils.data.DataLoader(mnist_test,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=1)
        last_epoch = -1
        if os.path.exists('./model.pt'):
            checkpoint = torch.load('./model.pt')
            last_epoch = checkpoint['last_epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])

        num_training_steps = len(train_iter) * self.epochs
        optimizer = torch.optim.Adam([{"params": self.model.parameters(),
                                       "initial_lr": self.learning_rate}])
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=300,
                                                    num_training_steps=num_training_steps,
                                                    num_cycles=2, last_epoch=last_epoch)
        lrs = []
        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(train_iter):
                x, y = x.to(self.device), y.to(self.device)
                loss, logits = self.model(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # 执行梯度下降
                scheduler.step()
                lrs.append(scheduler.get_last_lr())
                if i % 50 == 0:
                    acc = (logits.argmax(1) == y).float().mean()
                    print("### Epochs [{}/{}]---batch[{}/{}]---acc {:.4}---loss {:.4}".format(
                        epoch, self.epochs, len(mnist_train) // self.batch_size, i, acc, loss.item()))

            print("### Epochs [{}/{}]--acc on test {:.4}".format(epoch, self.epochs,
                                                                 self.evaluate(test_iter, self.model, self.device)))
        torch.save({'last_epoch': scheduler.last_eopch,
                    'model_state_dict': self.model.state_dict()},
                   './model.pt')

        plt.figure(figsize=(7, 4))
        plt.plot(range(num_training_steps), lrs, label='lr')
        plt.legend(fontsize=13)
        plt.show()

    @staticmethod
    def evaluate(data_iter, net, device):
        net.eval()
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            for x, y in data_iter:
                x, y = x.to(device), y.to(device)
                logits = net(x)
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)
            net.train()
            return acc_sum / n


if __name__ == '__main__':
    model = MyModel()
    model.train()
