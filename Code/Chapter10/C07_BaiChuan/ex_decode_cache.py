"""
文件名: Code/Chapter05/C01_ConfigManage/E01_Params.py
创建时间: 2023/12/15
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
from Baichuan2_7B_Chat.configuration_baichuan import BaichuanConfig
from Baichuan2_7B_Chat.modeling_baichuan import BaichuanModel

if __name__ == '__main__':
    config = BaichuanConfig.from_pretrained('./Baichuan2_7B_Chat')
    model = BaichuanModel(config)
    past_key_values = None
    for i in range(5):
        print(f"第{i + 1}个时刻输出：")
        inp = torch.randint(0, 100, [1, 1])
        result = model(inp, past_key_values=past_key_values)
        print(f"last_hidden_state的形状: {result.last_hidden_state.shape}")  # [batch_size, seq_len, hidden_size]
        past_key_values = result.past_key_values
        print(f"len(past_key_values): {len(past_key_values)}")  # 有多少层 = 4
        print(f"len(past_key_values[0]: {len(past_key_values[0])}")  # 每层几个元素 = 2
        print(f"past_key_values[0][0].shape: {past_key_values[0][0].shape}")  # 每个key的形状

# 第1个时刻输出：
# last_hidden_state的形状: torch.Size([1, 1, 1002])
# len(past_key_values): 4
# len(past_key_values[0]: 2
# past_key_values[0][0].shape: torch.Size([1, 3, 1, 334])
# 第2个时刻输出：
# last_hidden_state的形状: torch.Size([1, 1, 1002])
# len(past_key_values): 4
# len(past_key_values[0]: 2
# past_key_values[0][0].shape: torch.Size([1, 3, 2, 334])
# 第3个时刻输出：
# last_hidden_state的形状: torch.Size([1, 1, 1002])
# len(past_key_values): 4
# len(past_key_values[0]: 2
# past_key_values[0][0].shape: torch.Size([1, 3, 3, 334])
# 第4个时刻输出：
# last_hidden_state的形状: torch.Size([1, 1, 1002])
# len(past_key_values): 4
# len(past_key_values[0]: 2
# past_key_values[0][0].shape: torch.Size([1, 3, 4, 334])
# 第5个时刻输出：
# last_hidden_state的形状: torch.Size([1, 1, 1002])
# len(past_key_values): 4
# len(past_key_values[0]: 2
# past_key_values[0][0].shape: torch.Size([1, 3, 5, 334])
