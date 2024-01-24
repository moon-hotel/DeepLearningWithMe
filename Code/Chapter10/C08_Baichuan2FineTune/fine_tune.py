import os
import math
import pathlib
from typing import Optional, Dict
from dataclasses import dataclass, field
import json

import torch
from torch.utils.data import Dataset
import transformers
from transformers.training_args import TrainingArguments

from Baichuan2_7B_Base.tokenization_baichuan import BaichuanTokenizer
from Baichuan2_7B_Base.configuration_baichuan import BaichuanConfig
from Baichuan2_7B_Base.modeling_baichuan import GenerationConfig
from Baichuan2_7B_Base.modeling_baichuan import BaichuanForCausalLM
from Baichuan2_7B_Base.modeling_baichuan import BaichuanModel


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Baichuan2_7B_Base")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            data_path,
            tokenizer,
            model_max_length,
            user_tokens=[195],  # <reserved_106> 区分是来自用户的文本
            assistant_tokens=[196],  # <reserved_107> 区分是来自助手回答的文本，即希望模型根据用户输入的文本生成的文本
    ):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))  # 载入原始所有数据，返回的是一个列表，即每个元素为一个样本（包含有多轮的上下文对话）
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length  # 上下文窗口的最大长度，默认为512
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        self.ignore_index = -100  # 序号忽略token的ID
        item = self.preprocessing(self.data[0])  # 处理第1个样本，即取出第1个对话内容：
        """
        { "id": "77771","conversations": [
            {"from": "human", "value": "请给出两句苏轼词中主题是中秋的句子\n"},
            {"from": "gpt","value": "好的，以下是你要求的内容：明月几时有？把酒问青天。\n不止天上宫阙，今夕是何年。"},
            {"from": "human", "value": "这首词是苏轼什么时候写的？\n"},
            {"from": "gpt","value": "这首词作于宋神宗熙宁九年（1076）年，即丙辰年的，中秋佳节。"}]
        }
        """
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["labels"]:
            if id_ == -100:
                continue
            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        """
        每次处理一个样本
        以下注释内容下面的样本为例进行注释：
        { "id": "77771","conversations": [
            {"from": "human", "value": "请给出两句苏轼词中主题是中秋的句子\n"},
            {"from": "gpt","value": "好的，以下是你要求的内容：明月几时有？把酒问青天。\n不止天上宫阙，今夕是何年。"},
            {"from": "human", "value": "这首词是苏轼什么时候写的？\n"},
            {"from": "gpt","value": "这首词作于宋神宗熙宁九年（1076）年，即丙辰年的，中秋佳节。"}]
        }
        :param example: 输入为一个字典，即一个包含有多轮对话的样本
        :return:
        """
        input_ids = []
        labels = []
        for message in example["conversations"]:  # 开始取对话中的每一句
            from_ = message["from"]
            value = message["value"]
            value_ids = self.tokenizer.encode(value)  # 先将文本进行tokenize转换成词表索引id
            if from_ == "human":  # 如果是用户输入的语句
                # 交替将所有human和assistant的内容拼接在一起，并且用 user_tokens 进行分隔
                input_ids += self.user_tokens + value_ids  # 则在token_id前面加上 user_tokens 的标识
                labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(value_ids)
            else:
                # 交替将所有human和assistant的内容拼接在一起，并且用 assistant_tokens 进行分隔
                input_ids += self.assistant_tokens + value_ids
                labels += [self.ignore_index] + value_ids

        # input_ids: [195, 92676, 19278, 48278, 26702, 93319, 92364, 73791, 10430, 82831, 5, 196, 2015, 65, 2835, 11024,
        #  1853, 8736, 70, 23387, 92855, 23656, 68, 89446, 92614, 79107, 66, 5, 5380, 24616, 93660, 96261,
        #  65, 92731, 94404, 84465, 92381, 66, 195, 17759, 93319, 92347, 26702, 11090, 15473, 68, 5, 196,
        #  17759, 93319, 92400, 92441, 93849, 92786, 93676, 94859, 93151, 31506, 97923, 92336, 92335, 92383,
        #  92373, 97905, 92381, 65, 92813, 94893, 94459, 2537, 65, 10430, 26231, 66]

        # ['<reserved_106>', '请', '给出', '两句', '苏轼', '词', '中', '主题是', '中秋', '的句子', '\n', '<reserved_107>',
        # '好的', '，', '以下', '是你', '要求', '的内容', '：', '明月', '几', '时有', '？', '把酒', '问', '青天', '。', '\n',
        # '不知', '天上', '宫', '阙', '，', '今', '夕', '是何', '年', '。', '<reserved_106>', '这首', '词', '是', '苏轼',
        # '什么时候', '写的', '？', '\n', '<reserved_107>', '这首', '词', '作', '于', '宋', '神', '宗', '熙', '宁', '九年',
        # '（', '1', '0', '7', '6', '）', '年', '，', '即', '丙', '辰', '年的', '，', '中秋', '佳节', '。']

        # labels: [2, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 2015, 65, 2835, 11024, 1853,
        # 8736, 70, 23387, 92855, 23656, 68, 89446, 92614, 79107, 66, 5, 5380, 24616, 93660, 96261, 65, 92731, 94404,
        # 84465, 92381, 66, 2, -100, -100, -100, -100, -100, -100, -100, -100, -100, 17759, 93319, 92400, 92441, 93849,
        # 92786, 93676, 94859, 93151, 31506, 97923, 92336, 92335, 92383, 92373, 97905, 92381, 65, 92813, 94893, 94459,
        # 2537, 65, 10430, 26231, 66]
        # labels to text: ['</s>', '<->', '<->', '<->', '<->', '<->', '<->', '<->', '<->', '<->', '<->', '<->',
        # '好的', '，', '以下', '是你', '要求', '的内容', '：', '明月', '几', '时有', '？', '把酒', '问', '青天', '。', '\n',
        # '不知', '天上', '宫', '阙', '，', '今', '夕', '是何', '年', '。', '</s>', '<->', '<->', '<->', '<->', '<->', '<->',
        # '<->', '<->', '<->', '这首', '词', '作', '于', '宋', '神', '宗', '熙', '宁', '九年', '（', '1', '0', '7', '6', '）',
        # '年', '，', '即', '丙', '辰', '年的', '，', '中秋', '佳节', '。']
        #
        # 注意: 上面的 <-> 是掌柜自己将-100随意指定的一个符号
        input_ids.append(self.tokenizer.eos_token_id)  # 加入结束token
        labels.append(self.tokenizer.eos_token_id)
        input_ids = input_ids[: self.model_max_length]  # 截取最大窗口长度
        labels = labels[: self.model_max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))  # padding
        labels += [self.ignore_index] * (self.model_max_length - len(labels))  #
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # model_args 中只有 model_name_or_path 这一个参数
    # data_args 中只有 data_path 这一个参数
    model = BaichuanForCausalLM.from_pretrained(  # 载入预训练模型
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir)  # cache_dir 用于缓存下载的代码及预训练模型
    # config = BaichuanConfig.from_pretrained(model_args.model_name_or_path)
    # model = BaichuanForCausalLM(config)
    tokenizer = BaichuanTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    if training_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["W_pack"],
            inference_mode=False,
            r=1,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    dataset = SupervisedDataset(
        data_args.data_path, tokenizer, training_args.model_max_length
    )  # 构造数据集
    trainer = transformers.Trainer(model=model, args=training_args,
                                   train_dataset=dataset, tokenizer=tokenizer)
    trainer.train()
    trainer.save_state()  # 保存整个trainer状态
    trainer.save_model(output_dir=training_args.output_dir)  # 仅保存模型权重


if __name__ == "__main__":
    train()
