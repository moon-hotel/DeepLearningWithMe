import torch
from Baichuan2_7B_Chat.tokenization_baichuan import BaichuanTokenizer
from Baichuan2_7B_Chat.modeling_baichuan import BaichuanForCausalLM
from transformers.generation.utils import GenerationConfig

tokenizer = BaichuanTokenizer.from_pretrained("./Baichuan2_7B_Chat", use_fast=False, trust_remote_code=True)
model = BaichuanForCausalLM.from_pretrained("./Baichuan2_7B_Chat", device_map="auto",
                                            torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("./Baichuan2_7B_Chat")
for k, v in model.generation_config.__dict__.items():
    print(f"{k} = {v}")
messages = []
messages.append({"role": "user", "content": "解释一下“温故而知新”"})
print(messages)
response = model.chat(tokenizer, messages, stream=True)
print(response)
