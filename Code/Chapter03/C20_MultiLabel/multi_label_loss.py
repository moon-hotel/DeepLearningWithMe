import torch
import torch.nn as nn
import numpy as np


# Sigmoid损失
def Sigmoid_loss(y_true, y_pred):
    loss = nn.MultiLabelSoftMarginLoss(reduction='mean')
    print(loss(y_pred, y_true))  # 0.5927


def prediction(logits, K):
    y_pred = np.argsort(-logits, axis=-1)[:, :K]
    print("预测标签：", y_pred)
    p = np.vstack([logits[r, c] for r, c in enumerate(y_pred)])
    print("预测概率：", p)


if __name__ == '__main__':
    y_true = torch.tensor([[1, 1, 0, 0], [0, 1, 0, 1]], dtype=torch.int16)
    y_pred = torch.tensor([[0.2, 0.5, 0, 0], [0.1, 0.5, 0, 0.8]], dtype=torch.float32)
    Sigmoid_loss(y_true, y_pred)
    prediction(y_pred, 2)
