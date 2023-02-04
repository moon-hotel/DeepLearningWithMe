"""
文件名: Code/Chapter03/C10_CrossEntropy/main.py
创建时间: 2023/2/2 20:15 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import torch.nn as nn
import torch

if __name__ == '__main__':
    logits = torch.tensor([[0.5, 0.3, 0.6], [0.5, 0.4, 0.3]])
    y = torch.LongTensor([2, 0])
    loss = nn.CrossEntropyLoss(reduction='mean')  # 返回的均值是除以的每一批样本的个数（不一定是batch_size）
    l = loss(logits, y)
    print(l)