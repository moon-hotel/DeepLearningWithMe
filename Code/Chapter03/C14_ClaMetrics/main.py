"""
文件名: Code/Chapter03/C14_ClaMetrics/main.py
创建时间: 2023/2/5 11:10 上午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from sklearn.metrics import classification_report

if __name__ == '__main__':
    y_true = [1, 1, 1, 0, 0, 0, 2, 2, 2, 2]
    y_pred = [1, 0, 0, 0, 2, 1, 0, 0, 2, 2]
    result = classification_report(y_true, y_pred,
                                   target_names=['class 0', 'class 1', 'class 2'])
    print(result)
