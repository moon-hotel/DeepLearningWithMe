import torch


def softmax(x):
    s = torch.exp(x)
    return s / torch.sum(s, dim=1, keepdim=True)


def crossEntropy(y_true, logits):
    c = -torch.log(logits.gather(1, y_true.reshape(-1, 1)))
    return torch.sum(c)


logits = torch.tensor([[0.5, 0.3, 0.6], [0.5, 0.4, 0.3]])
y = torch.LongTensor([2, 0])
c = crossEntropy(y, softmax(logits)) / len(y)
print(c)

loss = torch.nn.CrossEntropyLoss(reduction='mean')  # 返回的均值是除以的每一批样本的个数（不一定是batchsize）
cc = loss(logits, y)
print(cc)
