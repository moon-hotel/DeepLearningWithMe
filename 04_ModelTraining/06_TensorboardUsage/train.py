import torchvision
import torchvision.transforms as transforms
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import torch
from LeNet5 import LeNet5
import os
import tensorflow as tf
import tensorboard as tb
from copy import deepcopy

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']


def load_dataset(batch_size=64):
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
                 epochs=3,
                 learning_rate=0.01):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_save_path = 'model.pt'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = LeNet5()

    def train(self):
        train_iter, test_iter = load_dataset(self.batch_size)
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
        writer = SummaryWriter("runs/nin")
        self.model = self.model.to(self.device)
        max_test_acc = 0
        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(train_iter):
                x, y = x.to(self.device), y.to(self.device)
                loss, logits = self.model(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # 执行梯度下降
                scheduler.step()
                if i % 50 == 0:
                    acc = (logits.argmax(1) == y).float().mean()
                    print("### Epochs [{}/{}]---batch[{}/{}]---acc {:.4}---loss {:.4}".format(
                        epoch + 1, self.epochs, i, len(train_iter), acc, loss.item()))
                    writer.add_scalar('Training/Accuracy', acc, scheduler.last_epoch)
                writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
                writer.add_scalar('Training/Learning Rate', scheduler.get_last_lr()[0], scheduler.last_epoch)

            test_acc, all_logits, y_labels, label_img = self.evaluate(test_iter, self.model, self.device)
            print("### Epochs [{}/{}]--Acc on test {:.4}".format(epoch + 1, self.epochs, test_acc))
            writer.add_scalar('Testing/Accuracy', test_acc, scheduler.last_epoch)
            writer.add_embedding(mat=all_logits,  # 所有点
                                 metadata=y_labels,  # 标签名称
                                 label_img=label_img,  # 标签图片
                                 global_step=scheduler.last_epoch)
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                state_dict = deepcopy(self.model.state_dict())
            torch.save({'last_epoch': scheduler.last_epoch,
                        'model_state_dict': state_dict},
                       './model.pt')

    @staticmethod
    def evaluate(data_iter, net, device):
        net.eval()
        all_logits = []
        y_labels = []
        images = []
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            for x, y in data_iter:
                x, y = x.to(device), y.to(device)
                logits = net(x)
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)

                all_logits.append(logits)
                y_pred = logits.argmax(1).view(-1)
                y_labels += (text_labels[i] for i in y_pred)
                images.append(x)
            net.train()
            return acc_sum / n, torch.cat(all_logits, dim=0), y_labels, torch.cat(images, dim=0)


if __name__ == '__main__':
    model = MyModel()
    model.train()
