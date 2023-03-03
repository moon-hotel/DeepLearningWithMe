"""
文件名: Code/Chapter05/C01_ConfigManage/E01_Params.py
创建时间: 2023/2/26 3:47 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import os


def train(train_file_path=os.path.join('data', 'train.txt'),
          val_file_path=os.path.join('data', 'val.txt'),
          test_file_path=os.path.join('data', 'test.txt'),
          split_sep='_!_',
          is_sample_shuffle=True,
          batch_size=16,
          learning_rate=3.5e-5,
          max_sen_len=None,
          num_labels=3,
          epochs=5):
    dataset = get_dataset(train_file_path, val_file_path,max_sen_len,
                          test_file_path, split_sep, is_sample_shuffle)
    model = get_model(max_sen_len, num_labels)


if __name__ == '__main__':
    train()
