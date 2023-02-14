"""
文件名: Code/Chapter03/C18_HyperParams/main.py
创建时间: 2023/2/12 4:48 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
import torch.nn as nn
import numpy as np


def evaluate(data_iter, net):
    net.eval()
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        logits = net(x)
        acc_sum += (logits.argmax(1) == y).float().sum().item()
        n += len(y)
    return round(acc_sum / n, 4)


def load_dataset():
    data_train = MNIST(root='~/Datasets/MNIST', train=True, download=True
                       , transform=transforms.ToTensor())
    data_test = MNIST(root='~/Datasets/MNIST', train=False, download=True,
                      transform=transforms.ToTensor())
    return data_train, data_test


def get_model(input_node=28 * 28,
              hidden_nodes=1024,
              hidden_layers=0,
              output_node=10,
              p=0.5):
    net = nn.Sequential(nn.Flatten())
    for i in range(hidden_layers):
        net.append(nn.Linear(input_node, hidden_nodes))
        net.append(nn.Dropout(p=p))
        input_node = hidden_nodes
    net.append(nn.Linear(input_node, output_node))
    return net


def train(train_iter, val_iter, net, lr=0.03, weight_decay=0., epochs=1):
    net.train()
    loss = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)  # 定义优化器
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_iter):
            logits = net(x)
            l = loss(logits, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()  # 执行梯度下降
    return evaluate(train_iter, net), evaluate(val_iter, net)


def cross_validation(data_train,
                     k=2,
                     batch_size=128,
                     input_node=28 * 28,
                     hidden_nodes=1024,
                     hidden_layers=0,
                     output_node=10,
                     p=0.5,
                     weight_decay=0.,
                     lr=0.03):
    model = get_model(input_node, hidden_nodes, hidden_layers, output_node, p)
    kf = KFold(n_splits=k)
    val_acc_his = []
    for i, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(data_train)))):
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_iter = DataLoader(data_train, batch_size=batch_size, sampler=train_sampler)
        val_iter = DataLoader(data_train, batch_size=batch_size, sampler=val_sampler)
        train_acc, val_acc = train(train_iter, val_iter, model, lr, weight_decay)
        val_acc_his.append(val_acc)
        print(f"  # Fold {i} train acc: {train_acc}, val acc: {val_acc} finished.")
    return np.mean(val_acc_his), model


if __name__ == '__main__':
    data_train, data_test = load_dataset()
    k = 3
    batch_size = 128
    input_node = 28 * 28
    hidden_nodes = 1024
    output_node = 10

    hyp_hidden_layers = [0, 2]
    hyp_p = [0.5, 0.7]
    hyp_weight_decay = [0., 0.01]
    hyp_lr = [0.01, 0.03]
    best_val_acc = 0
    best_model = None
    best_params = None
    total_models = len(hyp_hidden_layers) * len(hyp_p) * len(hyp_weight_decay) * len(hyp_lr)
    print(f"### Total model {total_models}, "
          f"fitting times {k * total_models}")
    no_model = 1
    for hidden_layer in hyp_hidden_layers:
        for p in hyp_p:
            for weight_decay in hyp_weight_decay:
                for lr in hyp_lr:
                    print(f" ## Fitting model [{no_model}/{total_models}]......")
                    no_model += 1
                    mean_val_acc, model = cross_validation(data_train=data_train,
                                                           k=k, batch_size=batch_size,
                                                           input_node=input_node,
                                                           hidden_nodes=hidden_nodes,
                                                           hidden_layers=hidden_layer,
                                                           output_node=output_node,
                                                           p=p,
                                                           weight_decay=weight_decay,
                                                           lr=lr)
                    params = {"hidden_layer": hidden_layer, "p": p,
                              "weight_decay": weight_decay, "lr": lr,
                              "mean_val_acc": mean_val_acc}
                    if mean_val_acc > best_val_acc:
                        best_val_acc = mean_val_acc
                        best_model = model
                        best_params = params
                    print(f"{params}\n")

    test_iter = DataLoader(data_test, batch_size=128)
    print(f"The best model params: {best_params},"
          f"acc on test: {evaluate(test_iter, best_model)}")

    # ### Total model 16, fitting times 48
    #  ## Fitting model [1/16]......
    #   # Fold 0 train acc: 0.824, val acc: 0.8194 finished.
    #   # Fold 1 train acc: 0.8464, val acc: 0.8415 finished.
    #   # Fold 2 train acc: 0.8565, val acc: 0.8618 finished.
    # {'hidden_layer': 0, 'p': 0.5, 'weight_decay': 0.0, 'lr': 0.01, 'mean_val_acc': 0.8409}
    #
    #  ## Fitting model [2/16]......
    #   # Fold 0 train acc: 0.857, val acc: 0.8558 finished.
    #   # Fold 1 train acc: 0.8761, val acc: 0.8699 finished.
    #   # Fold 2 train acc: 0.8827, val acc: 0.8864 finished.
    # {'hidden_layer': 0, 'p': 0.5, 'weight_decay': 0.0, 'lr': 0.03, 'mean_val_acc': 0.8706999999999999}
    #
    #  ## Fitting model [3/16]......
    #   # Fold 0 train acc: 0.8238, val acc: 0.8184 finished.
    #   # Fold 1 train acc: 0.8477, val acc: 0.8419 finished.
    #   # Fold 2 train acc: 0.8572, val acc: 0.8642 finished.
    # {'hidden_layer': 0, 'p': 0.5, 'weight_decay': 0.01, 'lr': 0.01, 'mean_val_acc': 0.8414999999999999}
    #
    #  ## Fitting model [4/16]......
    #   # Fold 0 train acc: 0.8546, val acc: 0.8504 finished.
    #   # Fold 1 train acc: 0.8749, val acc: 0.8681 finished.
    #   # Fold 2 train acc: 0.8808, val acc: 0.8845 finished.
    # {'hidden_layer': 0, 'p': 0.5, 'weight_decay': 0.01, 'lr': 0.03, 'mean_val_acc': 0.8676666666666667}
    #
    #  ## Fitting model [5/16]......
    #   # Fold 0 train acc: 0.8234, val acc: 0.8171 finished.
    #   # Fold 1 train acc: 0.8479, val acc: 0.8428 finished.
    #   # Fold 2 train acc: 0.8563, val acc: 0.8637 finished.
    # {'hidden_layer': 0, 'p': 0.7, 'weight_decay': 0.0, 'lr': 0.01, 'mean_val_acc': 0.8412000000000001}
    #
    #  ## Fitting model [6/16]......
    #   # Fold 0 train acc: 0.8587, val acc: 0.8559 finished.
    #   # Fold 1 train acc: 0.8779, val acc: 0.8704 finished.
    #   # Fold 2 train acc: 0.8825, val acc: 0.8871 finished.
    # {'hidden_layer': 0, 'p': 0.7, 'weight_decay': 0.0, 'lr': 0.03, 'mean_val_acc': 0.8711333333333333}
    #
    #  ## Fitting model [7/16]......
    #   # Fold 0 train acc: 0.8209, val acc: 0.8158 finished.
    #   # Fold 1 train acc: 0.8483, val acc: 0.842 finished.
    #   # Fold 2 train acc: 0.856, val acc: 0.8619 finished.
    # {'hidden_layer': 0, 'p': 0.7, 'weight_decay': 0.01, 'lr': 0.01, 'mean_val_acc': 0.8399}
    #
    #  ## Fitting model [8/16]......
    #   # Fold 0 train acc: 0.8575, val acc: 0.8535 finished.
    #   # Fold 1 train acc: 0.8743, val acc: 0.8675 finished.
    #   # Fold 2 train acc: 0.8811, val acc: 0.8859 finished.
    # {'hidden_layer': 0, 'p': 0.7, 'weight_decay': 0.01, 'lr': 0.03, 'mean_val_acc': 0.8689666666666667}
    #
    #  ## Fitting model [9/16]......
    #   # Fold 0 train acc: 0.8007, val acc: 0.7947 finished.
    #   # Fold 1 train acc: 0.8532, val acc: 0.8488 finished.
    #   # Fold 2 train acc: 0.873, val acc: 0.8774 finished.
    # {'hidden_layer': 2, 'p': 0.5, 'weight_decay': 0.0, 'lr': 0.01, 'mean_val_acc': 0.8403}
    #
    #  ## Fitting model [10/16]......
    #   # Fold 0 train acc: 0.8748, val acc: 0.8692 finished.
    #   # Fold 1 train acc: 0.8958, val acc: 0.8905 finished.
    #   # Fold 2 train acc: 0.9014, val acc: 0.9024 finished.
    # {'hidden_layer': 2, 'p': 0.5, 'weight_decay': 0.0, 'lr': 0.03, 'mean_val_acc': 0.8873666666666667}
    #
    #  ## Fitting model [11/16]......
    #   # Fold 0 train acc: 0.7952, val acc: 0.7921 finished.
    #   # Fold 1 train acc: 0.8472, val acc: 0.8409 finished.
    #   # Fold 2 train acc: 0.8678, val acc: 0.8737 finished.
    # {'hidden_layer': 2, 'p': 0.5, 'weight_decay': 0.01, 'lr': 0.01, 'mean_val_acc': 0.8355666666666667}
    #
    #  ## Fitting model [12/16]......
    #   # Fold 0 train acc: 0.8702, val acc: 0.8656 finished.
    #   # Fold 1 train acc: 0.8915, val acc: 0.8871 finished.
    #   # Fold 2 train acc: 0.8956, val acc: 0.8976 finished.
    # {'hidden_layer': 2, 'p': 0.5, 'weight_decay': 0.01, 'lr': 0.03, 'mean_val_acc': 0.8834333333333332}
    #
    #  ## Fitting model [13/16]......
    #   # Fold 0 train acc: 0.7981, val acc: 0.7919 finished.
    #   # Fold 1 train acc: 0.8487, val acc: 0.8455 finished.
    #   # Fold 2 train acc: 0.8698, val acc: 0.875 finished.
    # {'hidden_layer': 2, 'p': 0.7, 'weight_decay': 0.0, 'lr': 0.01, 'mean_val_acc': 0.8374666666666667}
    #
    #  ## Fitting model [14/16]......
    #   # Fold 0 train acc: 0.869, val acc: 0.8654 finished.
    #   # Fold 1 train acc: 0.8933, val acc: 0.8876 finished.
    #   # Fold 2 train acc: 0.9003, val acc: 0.902 finished.
    # {'hidden_layer': 2, 'p': 0.7, 'weight_decay': 0.0, 'lr': 0.03, 'mean_val_acc': 0.8849999999999999}
    #
    #  ## Fitting model [15/16]......
    #   # Fold 0 train acc: 0.7978, val acc: 0.7917 finished.
    #   # Fold 1 train acc: 0.8489, val acc: 0.8434 finished.
    #   # Fold 2 train acc: 0.8677, val acc: 0.8731 finished.
    # {'hidden_layer': 2, 'p': 0.7, 'weight_decay': 0.01, 'lr': 0.01, 'mean_val_acc': 0.8360666666666666}
    #
    #  ## Fitting model [16/16]......
    #   # Fold 0 train acc: 0.8664, val acc: 0.8612 finished.
    #   # Fold 1 train acc: 0.8904, val acc: 0.8844 finished.
    #   # Fold 2 train acc: 0.8962, val acc: 0.8986 finished.
    # {'hidden_layer': 2, 'p': 0.7, 'weight_decay': 0.01, 'lr': 0.03, 'mean_val_acc': 0.8814000000000001}
    #
    # The best model params: {'hidden_layer': 2, 'p': 0.5, 'weight_decay': 0.0, 'lr': 0.03, 'mean_val_acc': 0.8873666666666667},acc on test: 0.9072
