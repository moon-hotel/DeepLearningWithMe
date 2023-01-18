import os
import torch


class Config():
    """
    基于Transformer架构的类Translation模型配置类
    """

    def __init__(self):
        #   数据集设置相关配置
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(self.project_dir, 'data')
        self.train_corpus_file_paths = os.path.join(self.dataset_dir, 'ag_news_csv', 'train.csv')
        self.test_corpus_file_paths = os.path.join(self.dataset_dir, 'ag_news_csv', 'test.csv')
        self.min_freq = 1
        self.max_sen_len = None

        #  模型相关配置
        self.batch_size = 128
        self.d_model = 512
        self.num_head = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dim_feedforward = 512
        self.dim_classification = 256
        self.num_class = 4
        self.dropout = 0.1
        self.concat_type = 'avg'
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.epsilon = 10e-9
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = 10
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.model_save_per_epoch = 2
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)


if __name__ == '__main__':
    config = Config()
    print(config.project_dir)
    print(config.train_corpus_file_paths)
