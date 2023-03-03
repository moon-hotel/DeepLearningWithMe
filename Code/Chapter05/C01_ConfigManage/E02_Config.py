"""
文件名: Code/Chapter05/C01_ConfigManage/E02_Config.py
创建时间: 2023/2/26 3:47 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import os


class ModelConfig(object):
    def __init__(self,
                 train_file_path=os.path.join('data', 'train.txt'),
                 val_file_path=os.path.join('data', 'val.txt'),
                 test_file_path=os.path.join('data', 'test.txt'),
                 split_sep='_!_',
                 is_sample_shuffle=True,
                 batch_size=16,
                 learning_rate=3.5e-5,
                 max_sen_len=None,
                 num_labels=3,
                 epochs=5):
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.test_file_path = test_file_path
        self.split_sep = split_sep
        self.is_sample_shuffle = is_sample_shuffle
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_sen_len = max_sen_len
        self.num_labels = num_labels
        self.epochs = epochs

#
def train(config):
    dataset = get_dataset(config)
    model = get_mode(config)


if __name__ == '__main__':
    config = ModelConfig(epochs=10)
    print(f"epochs = {config.epochs}")
    # train(config)
