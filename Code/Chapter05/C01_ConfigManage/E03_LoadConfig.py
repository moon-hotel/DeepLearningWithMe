"""
文件名: Code/Chapter05/C01_ConfigManage/E03_LoadConfig.py
创建时间: 2023/3/3 8:15 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import json
import six


class ModelConfig(object):
    def __init__(self, batch_size=16,
                 learning_rate=3.5e-5,
                 num_labels=3,
                 epochs=5):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        self.epochs = epochs

    @classmethod
    def from_json_file(cls, json_file):
        """从json配置文件读取配置信息"""
        with open(json_file, 'r') as reader:
            text = reader.read()
        model_config = ModelConfig()
        for (key, value) in dict(json.loads(text)).items():
            model_config.__dict__[key] = value
        return model_config


if __name__ == '__main__':
    config = ModelConfig.from_json_file("./config.json")
    print(config.hidden_dropout_prob)
    print(config.hidden_size)
