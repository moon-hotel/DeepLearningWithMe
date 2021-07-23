import torchvision
import torchvision.transforms as transforms
import torch
from LeNet6 import LeNet6
from LeNet6 import para_state_dict


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
    model = LeNet6()  # 初始化现有模型的权重参数
    print("初始化参数 conv.0.bias 为：", model.state_dict()['conv.0.bias'])
    model.to(device)
    state_dict = para_state_dict(model, model_save_dir)
    model.load_state_dict(state_dict)
    model.eval()
    print("载入本地模型重新初始化 conv.0.bias 为：", model.state_dict()['conv.0.bias'])
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += len(y)
        print("Accuracy in test data is :", acc_sum / n)


if __name__ == '__main__':
    train_iter, test_iter = load_dataset(64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference(test_iter, device)
