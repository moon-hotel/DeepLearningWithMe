"""
文件名: Code/Chapter09/C06_Seq2Seq/bleu_usage.py
创建时间: 2023/7/30 10:02 上午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
from torchtext.data.metrics import bleu_score


def example():
    # pred_seq = [['我', '是', '一位', '老师']]
    # label_seq = [[['我', '是', '一个', '学生']]]

    # pred_seq = [['我', '是', '一位', '老师']]
    # label_seq = [[['我', '是', '一个', '学生'], ['我', '是', '一位', '学生', '。']]]

    pred_seq = [['我', '是', '一位', '老师'], ['跟我', '学', '深度学习']]
    label_seq = [[['我', '是', '一个', '学生'], ['我', '是', '一位', '学生', '。']],
                 [['跟我', '一起', '学', '机器学习']]]

    max_n = 2
    bleu = bleu_score(pred_seq, label_seq, max_n, [1 / max_n] * max_n)
    return bleu


if __name__ == '__main__':
    bleu = example()
    print(bleu)
