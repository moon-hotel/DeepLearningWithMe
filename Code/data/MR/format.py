"""
文件名: Code/data/MR/format.py
创建时间: 2023/7/18 22:11 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
from sklearn.model_selection import train_test_split


def read_data(path):
    samples = []
    with open(path, encoding='iso-8859-1') as f:
        for line in f:
            samples.append(line.strip('\n'))
    if 'pos' in path:
        labels = [1] * len(samples)
    else:
        labels = [0] * len(samples)  #
    return samples, labels


def format_data():
    file_paths = ['rt-polarity.neg', 'rt-polarity.pos']
    all_samples, all_labels = [], []
    for path in file_paths:
        result = read_data(path)
        all_samples += result[0]
        all_labels += result[1]
    x_train, x_test, y_train, y_test = \
        train_test_split(all_samples, all_labels, test_size=0.3, random_state=10)

    with open('./rt_train.txt', 'w', encoding='utf-8') as f:
        for item in zip(x_train, y_train):
            f.write(item[0] + '_!_' + str(item[1]) + '\n')
    with open('./rt_val.txt', 'w', encoding='utf-8') as f:
        for item in zip(x_test, y_test):
            f.write(item[0] + '_!_' + str(item[1]) + '\n')
    with open('./rt_test.txt', 'w', encoding='utf-8') as f:
        for item in zip(x_test, y_test):
            f.write(item[0] + '_!_' + str(item[1]) + '\n')


if __name__ == '__main__':
    format_data()
