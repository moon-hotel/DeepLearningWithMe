"""
文件名: Code/Chapter10/C01_ELMo/ELMo.py
创建时间: 2023/8/27 10:09
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
from ELMo import ELMoCharacterCNN
from ELMo import ELMoBiLSTM
from ELMo import ELMoRepresentation
from ELMo import ELMoLM
import torch


class ModelConfig(object):
    def __init__(self):
        super().__init__()
        self.char_embed_num = 262
        self.char_embed_dim = 16
        self.max_characters_per_token = 50
        self.char_cnn_filters = [[1, 32], [2, 32], [3, 64],
                                 [4, 128], [5, 256], [6, 512], [7, 1024]]
        self.n_filters = sum(f[1] for f in self.char_cnn_filters)  # 2048
        self.projection_dim = 512
        self.n_layers = 2
        self.n_highway = 2
        self.vocab_size = 800
        self.charcnn_model = 'charcnn_model.pt'
        self.elmo_bilstm_model = 'elmo_bilstm_model.pt'
        self.freeze = False


def test_ELMoCharacterCNN():
    config = ModelConfig()
    model = ELMoCharacterCNN(config)
    token_ids = torch.randint(0, 100, [2, 6, 50])
    print(model(token_ids).shape)  # [batch_size, seq_len, projection_dim]
    torch.save(model.state_dict(), "charcnn_model.pt")


def test_ELMoBiLSTM():
    config = ModelConfig()
    token_embedding = torch.rand([2, 6, config.projection_dim])
    model = ELMoBiLSTM(config)
    for i in range(config.n_layers + 1):
        print("layer: ", i, model(token_embedding)[i].shape)
        # ([batch_size, seq_len, projection_dim])
    torch.save(model.state_dict(), "elmo_bilstm_model.pt")


def test_ELMoLM():
    config = ModelConfig()
    token_ids = torch.randint(0, 100, [2, 6, 50])
    y = torch.randint(0, config.vocab_size, [2, 6])
    # model = ELMoLM(config)
    # torch.save(model.state_dict(), "model.pt")
    model = ELMoLM.from_pretrained(config, "elmolm_model.pt", freeze=True)
    for (name, param) in model.named_parameters():
        print(param.requires_grad)
    loss = model(token_ids, y)
    print(loss)


def test_ELMoRepresentation():
    config = ModelConfig()
    token_ids = torch.randint(0, 100, [2, 6, 50])
    model = ELMoRepresentation(config)
    outputs, elmo_rep = model(token_ids)
    print(outputs.shape)  # [n_layers+1, batch_size, seq_len, projection_dim*2]
    print(elmo_rep.shape)  # [batch_size, seq_len, projection_dim*2]


if __name__ == '__main__':
    test_ELMoCharacterCNN()
    test_ELMoBiLSTM()
    # test_ELMoLM()
    test_ELMoRepresentation()
