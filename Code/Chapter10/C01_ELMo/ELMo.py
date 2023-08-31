"""
文件名: Code/Chapter10/C01_ELMo/ELMo.py
创建时间: 2023/8/26 21:16 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import torch
import torch.nn as nn
import logging
import os


# {"lstm": {"use_skip_connections": true, "projection_dim": 512, "cell_clip": 3, "proj_clip": 3, "dim": 4096,
#           "n_layers": 2},
#  "char_cnn": {"activation": "relu", "filters": [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]],
#               "n_highway": 2, "embedding": {"dim": 16}, "n_characters": 262, "max_characters_per_token": 50}}

class ModelConfig(object):
    def __init__(self):
        super().__init__()
        self.char_embed_num = 262
        self.char_embed_dim = 16
        self.max_characters_per_token = 50
        self.char_cnn_filters = [[1, 32], [2, 32], [3, 64],
                                 [4, 128], [5, 256], [6, 512], [7, 1024]]
        self.n_filters = sum(f[1] for f in self.char_cnn_filters)
        self.projection_dim = 512
        self.n_layers = 2
        self.n_highway = 2


class HighWay(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.highway = nn.Linear(config.n_filters, config.n_filters * 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_state):
        """
        :param hidden_state:  [batch_size, hidden_size]
        :return: [batch_size, hidden_size]
        """
        highway = self.highway(hidden_state)  # 分别计算后面的非线性部分和门 [batch_size * seq_len, n_filters*2]
        nonlinear_part, gate = highway.chunk(2, dim=-1)  # 两部分的形状均为 [batch_size * seq_len, n_filters]
        nonlinear_part = self.relu(nonlinear_part)  # [batch_size * seq_len, n_filters]
        gate = self.sigmoid(gate)  # [batch_size * seq_len, n_filters]
        hidden_state = gate * hidden_state + (1 - gate) * nonlinear_part
        return hidden_state


class PretrainedModel(nn.Module):
    def __init__(self, ):
        super(PretrainedModel, self).__init__()
        pass

    @classmethod
    def from_pretrained(cls, config=None, pretrained_model_path=None):
        model = cls(config)
        if not os.path.exists(pretrained_model_path):
            raise ValueError(f"模型{pretrained_model_path}不存在")
        loaded_paras = torch.load(pretrained_model_path)
        model.load_state_dict(loaded_paras)
        print(f"成功载入预训练模型{pretrained_model_path}")
        if config.freeze:
            for (name, param) in model.named_parameters():
                # logging.info(f"冻结参数{name}")
                print(f"冻结参数{name}")
                param.requires_grad = False
        return model


class ELMoCharacterCNN(PretrainedModel):
    """
    ELMo Character CNN  得到每个单词的token embedding
    """

    def __init__(self, config=None):
        super(ELMoCharacterCNN, self).__init__()
        self.config = config
        self.char_embedding = nn.Embedding(config.char_embed_num,
                                           config.char_embed_dim)
        conv = []
        # char_cnn_filters: [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]]
        for i, (width, num) in enumerate(config.char_cnn_filters):
            conv.append(nn.Conv1d(in_channels=config.char_embed_dim,
                                  out_channels=num, kernel_size=width, bias=True))
        self.char_conv = nn.ModuleList(conv)
        self.relu = nn.ReLU()
        self.highway = nn.ModuleList([HighWay(config) for _ in range(config.n_highway)])
        self.projection = nn.Linear(config.n_filters, config.projection_dim)

    def forward(self, x):
        """
        :param x: shape: [batch_size, seq_len, max_chars_per_token]
        :return: token_embedding: shape: [batch_size, seq_len, projection_dim]
        """
        seq_len = x.shape[1]
        x = self.char_embedding(x)  # [batch_size, seq_len, max_chars_per_token, char_embed_dim]
        x = x.reshape(-1, x.shape[2], x.shape[3])  # [batch_size*seq_len, max_chars_per_token, char_embed_dim]
        x = x.transpose(1, 2)  # [batch_size*seq_len, char_embed_dim, max_chars_per_token]
        convs = []
        for conv in self.char_conv:
            convolved = conv(x)  # [batch_size*seq_len, n_filters_of_each_cnn, max_chars_per_token - width + 1]
            convolved, _ = torch.max(convolved, dim=-1)  # 在最后一个维度，即特征通道这个维度上取最大池化，
            convolved = self.relu(convolved)  # [batch_size*seq_len, n_filters_of_each_cnn]
            convs.append(convolved)
        token_embedding = torch.cat(convs, dim=-1)  # linear_part, [batch_size * seq_len, n_filters]
        # Highway
        for highway in self.highway:
            token_embedding = highway(token_embedding)  # [batch_size * seq_len, n_filters]
        token_embedding = self.projection(token_embedding)  # [batch_size * seq_len, projection_dim]
        token_embedding = token_embedding.reshape(-1, seq_len, self.config.projection_dim)
        return token_embedding  # [batch_size, seq_len, projection_dim]


class ELMoBiLSTM(PretrainedModel):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        forward_layers, back_layers = [], []
        for _ in range(config.n_layers):
            lstm_forward = nn.LSTM(input_size=config.projection_dim, hidden_size=config.projection_dim,
                                   num_layers=1, batch_first=True)
            lstm_backward = nn.LSTM(input_size=config.projection_dim, hidden_size=config.projection_dim,
                                    num_layers=1, batch_first=True)
            forward_layers.append(lstm_forward)
            back_layers.append(lstm_backward)
        self.forward_layers = nn.ModuleList(forward_layers)
        self.back_layers = nn.ModuleList(back_layers)

    def forward(self, x):
        """

        :param x:  shape: [batch_size, seq_len, projection_dim]
        :return: [([batch_size, seq_len, projection_dim*2]), ([batch_size, seq_len, projection_dim*2]),...]
        """
        forward_cache, backward_cache = x, x.flip(1)
        outputs = [torch.cat([forward_cache, backward_cache], dim=-1)]  # token embedding
        for layer_id in range(self.config.n_layers):
            forward_output = self.forward_layers[layer_id](forward_cache)[0]  # [batch_size, seq_len, projection_dim]
            backward_output = self.back_layers[layer_id](backward_cache)[0]  # [batch_size, seq_len, projection_dim]
            if layer_id != 0:  # skip-connection 第一层没有残差
                forward_output += forward_cache
                backward_output += backward_cache
            outputs.append(torch.cat([forward_output, backward_output], dim=-1))
            forward_cache = forward_output  # [batch_size, seq_len, projection_dim]
            backward_cache = backward_output  # [batch_size, seq_len, projection_dim]
        return outputs


class ELMoLM(PretrainedModel):
    def __init__(self, config=None):
        """
        :param config:
        """
        super().__init__()
        self.config = config
        self.char_cnn = ELMoCharacterCNN(config)
        self.lstm = ELMoBiLSTM(config)
        self.classifier = nn.Linear(config.projection_dim, config.vocab_size)

    def forward(self, x, labels=None):
        """
        需要把每个单词按字符进行tokenize
        :param x: [batch_size, seq_len, max_chars_per_token]
        :param labels: [batch_size, seq_len]
        :return:
        """
        char_embedding = self.char_cnn(x)  # # [batch_size, seq_len, projection_dim]
        outputs = self.lstm(char_embedding)[-1]  # 只取最后一层的输出
        f_logits = outputs[:, :, :self.config.projection_dim]  # [batch_size, seq_len, projection_dim]
        f_logits = self.classifier(f_logits)  # [batch_size, seq_len, vocab_size]
        b_logits = outputs[:, :, -self.config.projection_dim:]  # [batch_size, seq_len, projection_dim]
        b_logits = self.classifier(b_logits)  # [batch_size, seq_len, vocab_size]
        fn_loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
        f_loss = fn_loss(f_logits.reshape(-1, self.config.vocab_size), labels.reshape(-1)) / x.shape[0]
        b_loss = fn_loss(b_logits.reshape(-1, self.config.vocab_size), labels.flip(1).reshape(-1)) / x.shape[0]
        total_loss = f_loss + b_loss
        return total_loss


class ELMoRepresentation(nn.Module):
    def __init__(self, config=None, rep_weights=None):
        """
        :param config:
        :param rep_weight: 每一层的特征表示的权重，list长度必须为 config.n_layers + 1
        """
        super().__init__()
        self.config = config
        self.char_cnn = ELMoCharacterCNN.from_pretrained(config, config.charcnn_model)
        self.lstm = ELMoBiLSTM.from_pretrained(config, config.elmo_bilstm_model)
        rep_weights_shape = [config.n_layers + 1, 1, 1, 1]
        if rep_weights is None or len(rep_weights) != config.n_layers + 1:
            if rep_weights is not None and len(rep_weights) != config.n_layers + 1:
                logging.warning(f"rep_weights指定无效，其长度必须为{config.n_layers + 1}")
            self.rep_weights = nn.Parameter(torch.randn(rep_weights_shape))
        else:
            self.rep_weights = torch.tensor(rep_weights).reshape(rep_weights_shape)

    def forward(self, x):
        """
        需要把每个单词按字符进行tokenize
        :param x: [batch_size, seq_len, max_chars_per_token]
        :return:
        """
        char_embedding = self.char_cnn(x)  # # [batch_size, seq_len, projection_dim]
        outputs = self.lstm(char_embedding)
        # [([batch_size, seq_len, projection_dim*2]), ([batch_size, seq_len, projection_dim*2]),...]
        outputs = torch.stack(outputs, dim=0)  # [n_layers+1, batch_size, seq_len, projection_dim*2]
        elmo_rep = (outputs * self.rep_weights).sum(0)  # [batch_size, seq_len, projection_dim*2]
        return outputs, elmo_rep
