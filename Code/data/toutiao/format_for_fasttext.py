"""
文件名: Code/data/toutiao/format_for_fasttext.py
创建时间: 2023/7/23 09:23 上午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import sys

sys.path.append("../../")
from utils import tokenize

label_name = ['故事', '文化', '娱乐', '体育', '财经', '房产', '汽车', '教育',
              '科技', '军事', '旅游', '国际', '股票', '三农', '游戏']


def format(path=None):
    new_path = path.split(".")[0] + "_fasttext2.txt"
    file = open(new_path, 'w', encoding='utf-8')
    with open(path, encoding='utf-8') as f:
        for line in f:
            sen, label = line.strip('\n').split('_!_')
            sen = " ".join(tokenize(sen, cut_words=True))
            label = label_name[int(label)]
            file.write('_!_' + label + ' ' + sen + '\n')
    file.close()


if __name__ == '__main__':
    paths = ["toutiao_train.txt", "toutiao_val.txt", "toutiao_test.txt"]
    for path in paths:
        format(path)
