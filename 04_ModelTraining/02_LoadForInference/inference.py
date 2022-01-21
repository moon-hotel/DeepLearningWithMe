import torchvision
import torchvision.transforms as transforms
import torch
from LeNet5 import LeNet5
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


def inference(data_iter, device, model_save_dir='./MODEL'):
    net = LeNet5()  # 初始化现有模型的权重参数
    net.to(device)
    model_save_path = os.path.join(model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        net.load_state_dict(loaded_paras)  # 用本地已有模型来重新初始化网络权重参数
        net.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += len(y)
        print("Accuracy in test data is :", acc_sum / n)


if __name__ == '__main__':
    train_iter, test_iter = load_dataset(64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference(test_iter, device)
