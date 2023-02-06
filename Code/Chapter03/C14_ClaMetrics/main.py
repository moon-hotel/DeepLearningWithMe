"""
文件名: Code/Chapter03/C14_ClaMetrics/main.py
创建时间: 2023/2/5 11:10 上午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from sklearn.metrics import classification_report
import torch


def calculate_top_k_accuracy(logits, targets, k=2):
    """
    计算top-k准确率
    :param logits: shape: (m,n)
    :param targets:  shape: (m,)
    :param k:
    :return:
    """
    values, indices = torch.topk(logits, k=k, sorted=True)
    y = torch.reshape(targets, [-1, 1])
    correct = (y == indices) * 1.  # 对比预测的K个值中是否包含有正确标签中的结果
    top_k_accuracy = torch.mean(correct) * k  # 计算最后的准确率
    return top_k_accuracy


if __name__ == '__main__':
    y_true = [1, 1, 1, 0, 0, 0, 2, 2, 2, 2]
    y_pred = [1, 0, 0, 0, 2, 1, 0, 0, 2, 2]
    result = classification_report(y_true, y_pred,
                                   target_names=['class 0', 'class 1', 'class 2'])
    print(result)

    logits = torch.tensor([[0.1, 0.3, 0.2, 0.4],
                           [0.5, 0.01, 0.9, 0.4]])
    y = torch.tensor([3, 0])
    print(calculate_top_k_accuracy(logits, y, k=1).item())  # 0.5
    print(calculate_top_k_accuracy(logits, y, k=2).item())  # 1.0
