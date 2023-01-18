import os
import torch
from utils.log_helper import Logger
import logging
from torch.utils.tensorboard import SummaryWriter

class Config():
    """
    基于Transformer架构的类Translation模型配置类
    """

    def __init__(self):
        #   数据集设置相关配置
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'couplet')
        self.train_corpus_file_paths = [os.path.join(self.dataset_dir, 'train', 'in.txt'),  # 训练时编码器的输入
                                        os.path.join(self.dataset_dir, 'train', 'out.txt')]  # 训练时解码器的输入
        self.test_corpus_file_paths = [os.path.join(self.dataset_dir, 'test', 'in.txt'),
                                       os.path.join(self.dataset_dir, 'test', 'out.txt')]
        self.min_freq = 1  # 在构建词表的过程中滤掉词（字）频小于min_freq的词（字）

        #  模型相关配置
        self.batch_size = 512
        self.d_model = 512
        self.num_head = 8
        self.num_encoder_layers = 5
        self.num_decoder_layers = 5
        self.dim_feedforward = 1024
        self.dropout = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.epsilon = 10e-9
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = 100
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.train_info_per_batch = 30
        self.model_save_per_epoch = 2
        self.writer = SummaryWriter("runs")
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # 日志相关
        self.logger = Logger(log_file_name='log_train',
                             log_level=logging.DEBUG,
                             log_dir=self.model_save_dir).get_log()
