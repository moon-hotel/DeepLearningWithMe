import os
import torch
from utils.log_helper import logger_init
import logging

class Config():
    """
    基于Transformer架构的类Translation模型配置类
    """

    def __init__(self):
        #   数据集设置相关配置
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data')
        self.train_corpus_file_paths = [os.path.join(self.dataset_dir, 'train.de'),  # 训练时编码器的输入
                                        os.path.join(self.dataset_dir, 'train.en')]  # 训练时解码器的输入
        self.val_corpus_file_paths = [os.path.join(self.dataset_dir, 'val.de'),  # 验证时编码器的输入
                                      os.path.join(self.dataset_dir, 'val.en')]  # 验证时解码器的输入
        self.test_corpus_file_paths = [os.path.join(self.dataset_dir, 'test_2016_flickr.de'),
                                       os.path.join(self.dataset_dir, 'test_2016_flickr.en')]
        self.min_freq = 1  # 在构建词表的过程中滤掉词（字）频小于min_freq的词（字）

        #  模型相关配置
        self.batch_size = 128
        self.d_model = 512
        self.num_head = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dim_feedforward = 512
        self.dropout = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.epsilon = 10e-9
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = 10
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        # 日志相关
        logger_init(log_file_name='log_train',
                    log_level=logging.INFO,
                    log_dir=self.model_save_dir)