import torchvision
import torchvision.transforms as transforms
import torch
from LeNet6 import LeNet6
from LeNet6 import para_state_dict
import os


def load_dataset(batch_size):
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',
                                                    train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',
                                                   train=False, download=True,
                                                   transform=transforms.ToTensor())
    train_iter = torch.utils.data.DataLoader(mnist_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=1)
    test_iter = torch.utils.data.DataLoader(mnist_test,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=1)
    return train_iter, test_iter


class MyModel:
    def __init__(self,
                 batch_size=64,
                 epochs=5,
                 learning_rate=0.001,
                 model_save_dir='./MODEL'):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_save_dir = model_save_dir
        self.model = LeNet6()

    def train(self):
        train_iter, test_iter = load_dataset(self.batch_size)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        model_save_path = os.path.join(self.model_save_dir, 'model_new.pt')
        old_model = os.path.join(self.model_save_dir, 'model.pt')
        if os.path.exists(old_model):
            state_dict = para_state_dict(self.model, self.model_save_dir)
            self.model.load_state_dict(state_dict)
            print("#### 成功载入已有模型，进行追加训练...")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)  # 定义优化器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(train_iter):
                x, y = x.to(device), y.to(device)
                loss, logits = self.model(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # 执行梯度下降
                if i % 100 == 0:
                    acc = (logits.argmax(1) == y).float().mean()
                    print("Epochs[{}/{}]---batch[{}/{}]---acc {:.4}---loss {:.4}".format(
                        epoch, self.epochs, len(train_iter), i, acc, loss.item()))

            print("Epochs[{}/{}]--acc on test {:.4}".format(epoch, self.epochs,
                                                            self.evaluate(test_iter, self.model, device)))
            torch.save(self.model.state_dict(), model_save_path)

    @staticmethod
    def evaluate(data_iter, model, device):
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            for x, y in data_iter:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)
            return acc_sum / n


if __name__ == '__main__':
    model = MyModel()
    model.train()
