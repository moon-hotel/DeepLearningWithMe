"""
文件名: Code/Chapter10/C07_BaiChuan/main.py
创建时间: 2023/12/25 22:17 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
from Baichuan2_7B_Chat.tokenization_baichuan import BaichuanTokenizer
from Baichuan2_7B_Chat.configuration_baichuan import BaichuanConfig
from Baichuan2_7B_Chat.modeling_baichuan import GenerationConfig
from Baichuan2_7B_Chat.modeling_baichuan import BaichuanForCausalLM
from Baichuan2_7B_Chat.modeling_baichuan import BaichuanModel


def test_BaichuanTokenizer():
    tokenizer = BaichuanTokenizer.from_pretrained("./Baichuan2_7B_Chat")
    inputs = tokenizer(["解释一下“温故而知新”", "好人一生平安"], return_tensors="pt", padding=True)
    print(inputs)
    for i in[ 195,  5987,  2908, 92360, 92880, 93009, 26058, 92422, 92361,   196]:
        print(tokenizer.decode(i))
    print(tokenizer.decode(inputs['input_ids'][0])) # 温故而知新

def test_GenerationConfig():
    generation_config = GenerationConfig.from_pretrained("./Baichuan2_7B_Chat")
    for k,v in generation_config.__dict__.items():
        print(f" {k} = {v}")
    #  max_length = 20
    #  max_new_tokens = 2048
    #  min_length = 0
    #  min_new_tokens = None
    #  early_stopping = False
    #  max_time = None
    #  do_sample = True
    #  num_beams = 1
    #  num_beam_groups = 1
    #  penalty_alpha = None
    #  use_cache = True
    #  temperature = 0.3
    #  top_k = 5
    #  top_p = 0.85
    #  typical_p = 1.0
    #  epsilon_cutoff = 0.0
    #  eta_cutoff = 0.0
    #  diversity_penalty = 0.0
    #  repetition_penalty = 1.05
    #  encoder_repetition_penalty = 1.0
    #  length_penalty = 1.0
    #  no_repeat_ngram_size = 0
    #  bad_words_ids = None
    #  force_words_ids = None
    #  renormalize_logits = False
    #  constraints = None
    #  forced_bos_token_id = None
    #  forced_eos_token_id = None
    #  remove_invalid_values = False
    #  exponential_decay_length_penalty = None
    #  suppress_tokens = None
    #  begin_suppress_tokens = None
    #  forced_decoder_ids = None
    #  num_return_sequences = 1
    #  output_attentions = False
    #  output_hidden_states = False
    #  output_scores = False
    #  return_dict_in_generate = False
    #  pad_token_id = 0
    #  bos_token_id = 1
    #  eos_token_id = 2
    #  encoder_no_repeat_ngram_size = 0
    #  decoder_start_token_id = None
    #  generation_kwargs = {}
    #  _from_model_config = False
    #  _commit_hash = None
    #  transformers_version = 4.29.2
    #  user_token_id = 195
    #  assistant_token_id = 196

def test_BaichuanModel():
    config = BaichuanConfig.from_pretrained('./Baichuan2_7B_Chat')
    model = BaichuanModel(config)
    config.show_paras()
    print(config.output_attentions)
    past_key_values = None
    print(config.return_dict)
    inp = torch.randint(0, 100, [1, 3])
    for i in range(4,7):
        print(f"第{i}个时刻输出: ")
        result = model(inp, past_key_values=past_key_values)
        print(f"last_hidden_state的形状: {result.last_hidden_state.shape}")  # [batch_size, seq_len, hidden_size]
        past_key_values = result.past_key_values
        print(f"len(past_key_values): {len(past_key_values)}")  # 有多少层 = 4
        print(f"len(past_key_values[0]: {len(past_key_values[0])}")  # 每层几个元素 = 2
        print(f"past_key_values[0][0].shape: {past_key_values[0][0].shape}")  # 每个key的形状
        inp = torch.randint(0, 100, [1, 1])



if __name__ == '__main__':
    test_BaichuanTokenizer()
    # test_BaichuanModel()
    # test_GenerationConfig()