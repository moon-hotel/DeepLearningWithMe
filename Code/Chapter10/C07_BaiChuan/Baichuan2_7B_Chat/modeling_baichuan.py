# Copyright 2023 Baichuan Inc. All Rights Reserved.

# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .configuration_baichuan import BaichuanConfig
from .generation_utils import build_chat_input, TextIterStreamer

import math
from typing import List, Optional, Tuple, Union
from threading import Thread

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation.utils import GenerationConfig
from transformers.utils import logging, ContextManagers

import os
from contextlib import contextmanager

logger = logging.get_logger(__name__)

try:
    from xformers import ops as xops
except ImportError:
    xops = None
    logger.warning(
        "Xformers is not installed correctly. If you want to use memory_efficient_attention to accelerate training use the following command to install Xformers\npip install xformers."
    )

"""
特别说明，注释中的所有：
1. seq表示原始的输入序列，形状为[batch_size, seq_len]
2. seq_len表示原始输入序列的长度；
3. query_states, key_states, value_states表示本次前向传播时，seq经过3个线性变换得到的结果（后于用于自注意力计算）
   如果在推理过程中使用了use_cache，那么key_states, value_states后续还会拼接了上一时刻计算得到的key_states, value_states
4. query_len, key_len, value_len 表示本次前向传播时query_states, key_states, value_states序列的长度；
5. 在训练过程中 query_len == key_len == value_len == seq_len
6. 在推理过程中且use_cache=True， key_len == value_len, kv_seq_len = key_len + 上一时刻的key_len
"""


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    if len(mask.size()) == 3:
        bsz, src_len, _ = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, :, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    else:
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class RotaryEmbedding(torch.nn.Module):
    """
    旋转位置编码（Rotary Position Embedding，RoPE）是论文Roformer: Enhanced Transformer With Rotray Position Embedding
    提出的一种能够将相对位置信息依赖集成到 self-attention 中并提升 transformer 架构性能的位置编码方式。
    和相对位置编码相比，RoPE 具有更好的外推性，目前是大模型相对位置编码中应用最广的方式之一。
    外推性是指大模型在训练时和预测时的输入长度不一致，导致模型的泛化能力下降的问题。例如，如果一个模型在训练时只使用了512个 token 的文本，
    那么在预测时如果输入超过512个 token，模型可能无法正确处理。这就限制了大模型在处理长文本或多轮对话等任务时的效果。
    https://zhuanlan.zhihu.com/p/647109286
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :].to(torch.float32)
        self.sin_cached = emb.sin()[None, None, :, :].to(torch.float32)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :].to(torch.float32).to(x.device)
            self.sin_cached = emb.sin()[None, None, :, :].to(torch.float32).to(x.device)
        elif self.cos_cached.device != x.device:
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos_, sin_, position_ids):
    cos = cos_.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin_.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class MLP(nn.Module):
    """
    基于门机制的多层感知机
    """

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        # gate_proj: [batch_size, seq_len, intermediate_size]
        # up_proj(x): [batch_size, seq_len, intermediate_size]
        # return down_proj: [batch_size, seq_len, hidden_size]
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  #
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads  # 每个头的维度
        self.max_position_embeddings = config.max_position_embeddings
        # 位置编码支持的最大长度，配置文件中为4096，即支持上下文窗口4K

        if (self.head_dim * self.num_heads) != self.hidden_size:  # 需要能整除
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        # 对输入q,k,v同时进行线性变换，所以是3部分
        self.W_pack = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            # hidden_states: 第一次输入到Attention模块的是input_ids经过Token Embedding后的结果
            # 形状为[batch_size, seq_len, hidden_size]
            attention_mask: Optional[torch.Tensor] = None,
            # 注意力矩阵，用于在训练时掩盖当前时刻之后的信息，形状为[batch_size, 1, query_len, key_len]
            position_ids: Optional[torch.LongTensor] = None,  #
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            # 推理过程中才会用到, 用来传入截止上一个时刻解码时，之前所有时刻累计拼接得到的key_states和value_states
            # 关于该参数的作用，建议先从图示进行理解，past_key_value_1.jpg 和 past_key_value_2.jpg
            output_attentions: bool = False,
            # 是否返回注意力权重，默认为False，事实上本代码也不支持返回注意力权重矩阵，因为memory_efficient_attention和
            # scaled_dot_product_attention两个函数的返回结果都不包含注意力权重矩阵。且output_attentions=True时本代码还会报错
            # 算是一个bug， 因为在下面的代码过程attn_weights一开始没有声明，当output_attentions=True时报错:
            # UnboundLocalError:  local variable 'attn_weights' referenced before assignment
            use_cache: bool = False,  # 在推理工程中是否使用上一时刻计算的缓存结果加速，默认情况为使用
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()  # [batch_size, seq_len, hidden_size]

        # 根据输入同时计算得到query_states, key_states 和 value_states
        proj = self.W_pack(hidden_states)  # [batch_size, seq_len, 3 * hidden_size]
        proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)
        # proj: [batch_size, seq_len, 3 * hidden_size] ==> [batch_size, seq_len, 3, hidden_size]
        # ==> [1, batch_size, seq_len, 3, hidden_size] ==> [3, batch_size, seq_len, 1, hidden_size]
        # ==> [3, batch_size, seq_len, hidden_size]

        # 分离得到 query_states, key_states, value_states 三者形状相同
        # 在推理时使用 use_cache 时 query 除了解码第一个时刻时的输入是多个token，
        # 后续每次都只会将上一个时刻的输出作为下一个时刻的输入，即query_len = 1
        query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # query_states:  [batch_size, seq_len, hidden_size] ==> [batch_size, seq_len, num_heads, head_dim]
        # ==> [batch_size, num_heads, seq_len, head_dim]

        # past_key_value 不是None，表是使用了use_cache
        kv_seq_len = key_states.shape[-2]  #
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]  # 取上一时刻中key_states和query_states的序列长度
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)  # [1,1,kv_seq_len,head_dim]

        # 对当query_states, key_states 进行旋转位置编码
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [batch_size,num_heads, seq_len, head_dim]

        if past_key_value is not None:
            # reuse k, v, self_attention # 沿着序列长度维度拼接上一次的key和Value
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            # 上面两个形状均为: [batch_size,num_heads, past_key_len + seq_len, head_dim]
            # 后面我们还是用 kv_seq_len 来代指 key_states 和 value_states 的序列长度
            #    用query_len来代指query_states的序列长度，
            #    因为这里cat拼接以后key_states和value_states的序列长度会变长

        past_key_value = (key_states, value_states) if use_cache else None
        # 对本次解码时的key_states, value_states进行缓存，在下一个时刻进行复用
        if xops is not None and self.training:
            # 加速计算过程
            attn_weights = None
            query_states = query_states.transpose(1, 2)  # [batch_size, q_seq_len,num_heads, head_dim]
            key_states = key_states.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
            value_states = value_states.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
            attn_output = xops.memory_efficient_attention(
                query_states, key_states, value_states, attn_bias=xops.LowerTriangularMask()
            )
        else:
            # 采用Pytorch 2.0 版本的新特性，加速计算
            # query_states [batch_size,num_heads,query_len, head_dim]
            # key_states [batch_size,num_heads, kv_seq_len, head_dim]
            # value_states [batch_size,num_heads, kv_seq_len, head_dim]

            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states,
                                                             attn_mask=attention_mask)
                # print("attention_maskattention_mask", attention_mask.shape)
            # scaled_dot_product_attention 的计算过程
            #     attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))) + attn_mask, dim=-1)
            #           ==> [batch_size,num_heads,query_len, head_dim] @ [batch_size,num_heads, head_dim, kv_seq_len]
            #           ==> [batch_size,num_heads,query_len,kv_seq_len]  + [batch_size, 1, query_len,kv_seq_len]
            #     attn_weight = torch.dropout(attn_weight, dropout_p)
            #     return attn_weight @ V
            #           ==> [batch_size,num_heads,query_len,kv_seq_len] @ [batch_size,num_heads, kv_seq_len, head_dim]
            #           ==> [batch_size,num_heads,query_len,head_dim]
            attn_output = attn_output.transpose(1, 2)  # [batch_size, query_len, num_heads, head_dim]
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)  # [batch_size, query_len, hidden_size】
        attn_output = self.o_proj(attn_output)  # [batch_size, q_seq_len, hidden_size】

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class DecoderLayer(nn.Module):
    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config)
        self.mlp = MLP(hidden_size=self.hidden_size,
                       intermediate_size=config.intermediate_size,
                       hidden_act=config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states  # [batch_size, seq_len, hidden_size]

        hidden_states = self.input_layernorm(hidden_states)  # [batch_size, seq_len, hidden_size]

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache)
        hidden_states = residual + hidden_states  # [batch_size, seq_len, hidden_size]
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)  # [batch_size, seq_len, hidden_size]
        hidden_states = self.mlp(hidden_states)  # [batch_size, seq_len, hidden_size]
        hidden_states = residual + hidden_states  # [batch_size, seq_len, hidden_size]

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
        # if output_attentions and use_cache :
        #       outputs = (hidden_states, self_attn_weights, present_key_value)
        # if use_cache:
        #       outputs = (hidden_states, present_key_value)


class BaichuanPreTrainedModel(PreTrainedModel):
    config_class = BaichuanConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BaichuanModel):
            module.gradient_checkpointing = value


class BaichuanModel(BaichuanPreTrainedModel):
    def __init__(self, config: BaichuanConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 默认config.json中 padding_idx 为0
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """
        seq_length_with_past = seq_length
        seq_length_with_past = seq_length_with_past + past_key_values_length
        :param attention_mask: [batch_size, seq_length_with_past]
        :param input_shape: [batch_size, seq_length]
        :param inputs_embeds: [batch_size, seq_length, hidden_size]
        :param past_key_values_length:
        :return:
        """
        # 这个函数是用来构造 注意力掩码，输出形状为[batch_size, 1, query_len, key_len]
        # 当模型训练时，query_len == key_len, 即此时的attention_mask是一个正方形
        #
        # 当模型推理，且使用use_cache时，query为每次仅为上一时刻的输出，所以query_len=1， key_len会一直累加上一次key的长度
        #  此时得到的attention_mask的形状为[batch_size, 1, 1, key_len]，全是0，即不做 注意力遮蔽处理
        # create causal mask
        # [batch_size, seq_len] -> [batch_size, 1, query_len, key_len] # seq_len == key_len
        combined_attention_mask = None
        if input_shape[-1] > 1:
            # 有如下几种情况会走下面这段逻辑：
            # ① 模型训练时，此时 _make_causal_mask 返回的就是我们之前介绍过的一个方阵，如：
            #       tensor([[[[ 0.0000e+00, -3.4028e+38, -3.4028e+38],
            #                 [ 0.0000e+00,  0.0000e+00, -3.4028e+38],
            #                 [ 0.0000e+00,  0.0000e+00,  0.0000e+00]]]])
            #       torch.Size([1, 1, 3, 3]) [batch_size, 1, seq_len, seq_len]
            # ② 模型推理，且不使用 use_cache 时，
            #       首先第一个时刻解码时同样需要对输入的序列（长度大于1）进行编码，此时_make_causal_mask
            #       返回的也是一个类似上述方阵。且在后续解码过程中，由于没有使用use_cache,所以模型当前时刻
            #       的输入会拼接上之前所有时刻的输入，所以同样需要掩盖，返回的仍旧是一个类似上述方阵。
            # ③ 模型推理，且使用use_cache时，
            #       首先第一个时刻解码时同样需要对输入的序列（长度大于1）进行编码，此时_make_causal_mask
            #       返回的也是一个类似上述方阵。进一步，由于使用了 use_cache , 所以后续每个时刻模型的输入都只有上一个
            #       时刻的输出，即序列长度为1，所以不会进入本逻辑

            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length)

        if attention_mask is not None:
            # 此处的条件会始终成立，因为在 _prepare_decoder_attention_mask 函数前对attention_mask进行了处理
            # 传入进行来的
            # [batch_size, seq_length_with_past] -> [batch_size, 1, query_len, seq_length_with_past]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device)  # 全为0，用于当模型推理且使用use_cache时构造的注意力掩码

            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )  # 这里else 后面的 expanded_attn_mask 似乎没有用，
            # 因为 combined_attention_mask is not None的时候expanded_attn_mask全是0
        return combined_attention_mask
        # 总结：①模型训练或（模型推理且use_cache = False）时返回的就是我们之前介绍过的一个方阵，
        #        形状为[batch_size, 1, seq_len, seq_len]
        #      ②模型推理且use_cache = True 时，在编码用户第一次输入的序列时返回的也是一个方阵，
        #        后续解码每个时刻返回是一个全为0的结果，形状为 [1, 1, 1, seq_length_with_past]
        #           例如 输入模型： 乔峰是谁  模型在对其进行编码时将会构造一个[1,1,4,4]的attention_mask:
        #                         [[[[ 0.0000e+00, -3.3895e+38, -3.3895e+38, -3.3895e+38]
        #                            [0.0000e+00,  0.0000e+00, -3.3895e+38, -3.3895e+38]
        #                            [0.0000e+00,  0.0000e+00,  0.0000e+00, -3.3895e+38]
        #                            [0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]]]]
        #               后续模型在预测对每个时刻进行解码时，将会构造一个[1,1,1,seq_length_with_past] 的 attention_mask，且都为0
        #                    第6个时刻    [[[[0., 0., 0., 0., 0.]]]]
        #                    第7个时刻    [[[[0., 0., 0., 0., 0., 0.]]]]

    def forward(
            self,
            input_ids: torch.LongTensor = None,  # 原始输入索引id, 形状为 [batch_size, seq_len]
            attention_mask: Optional[torch.Tensor] = None,
            # attention mask, 即用于在训练阶段掩盖当前时刻t之后的信息（t+1,t+2,...）
            # 默认为空，不传入，后面会通过 _prepare_decoder_attention_mask 构造
            position_ids: Optional[torch.LongTensor] = None,  #
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            # 推理过程中才会用到, 用来传入截止上一个时刻解码时，之前所有时刻累计拼接得到的key_states和value_states
            # 关于该参数的作用，建议先从图示进行理解，past_key_value_1.jpg 和 past_key_value_2.jpg
            inputs_embeds: Optional[torch.FloatTensor] = None,
            # 直接传入embedding后的结果, input_ids将被忽略后续不再进行embedding, 形状为[batch_size, seq_len, hidden_size]
            use_cache: Optional[bool] = None,  # 是否使用key, value 缓存， 加快推理时的计算速度
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,  # 默认值为True
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # =============Step 1. 初始化相关控制参数 ==================
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 返回结果是否输出注意力 True or False，默认情况下为False
        # 这里的self.config是上面BaichuanModel()初始化时传入的config对象, 它一部分成员变量来自于本地的config.json文件
        # 一部分是config类继承自，BaichuanConfig和 Transformers中的PretrainedConfig类，
        # 包含有 output_attentions, output_hidden_states 等相关默认参数
        # 如果需要输出可在config.json中添加一行 "output_attentions": true
        # 不过事实上本代码也不支持返回注意力权重矩阵，因为memory_efficient_attention和
        # caled_dot_product_attention两个函数的返回结果都不包含注意力权重矩阵。
        output_hidden_states = (output_hidden_states if output_hidden_states is not None
                                else self.config.output_hidden_states)  # 默认为False
        use_cache = use_cache if use_cache is not None else self.config.use_cache  # 默认为 True

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 默认为 False

        # =============Step 2. 获取相关形状参数 ==================
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]  # 得到上一轮的key value序列的长度
            seq_length_with_past = seq_length_with_past + past_key_values_length
            # 本次当前时刻和上一时刻key, value拼接后的长度

        # =============Step 3. 构造模型相关输入 ==================
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length,
                                        dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            # 这里需要注意的是, position_ids 的起始位置为 past_key_values_length，也就是说如果传入了past_key_values
            # 那么past_key_values_length的起始值为上一个时刻key的长度, 例如: 如果past_key_values_length = 5
            # seq_length = 4, 则 position_ids 可能是 [5,6,7,8]
            # 在推理过程中，如果use_cache=True，那么在对输入进行编码时 position_ids 的形状为 [1, seq_len]，例如 [[0,1,2,3]]
            #             在后续逐时刻解码生成内容时，position_ids 的形状为 [1, 1], 第5个时刻为[[4]]， 第6个时刻为[[5]]
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        print("验证use_cache=Fasle时position_ids，和attention_mask，use_cache=",use_cache)
        print("position_ids ", position_ids.shape)
        print("position_ids ", position_ids)
        if inputs_embeds is None:
            # 传入inputs_embeds后，便不在进行embedding，例如传入经过第三方词向量嵌入后的表示
            inputs_embeds = self.embed_tokens(input_ids)  # [batch_size, seq_len, hidden_size]

        if attention_mask is None:  # 下面开始构造注意力掩码
            attention_mask = torch.ones((batch_size, seq_length_with_past),
                                        dtype=torch.bool, device=inputs_embeds.device)
            # 初始化attention_mask [batch_size, seq_length_with_past]
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
        # [batch_size, 1, seq_len, seq_len] 或 [1, 1, 1, seq_length_with_past]
        print("attention_mask", attention_mask)
        print("attention_mask", attention_mask.shape)
        print("============")
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with "
                                    "gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        # =============Step 4. 多层编码器前向传播过程 ==================
        # decoder layers
        all_hidden_states = () if output_hidden_states else None  # 默认为 False
        all_self_attns = () if output_attentions else None  # 默认为 False
        next_decoder_cache = () if use_cache else None  # 默认为 True

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # 存储每一层的隐藏状态， 默认不返回

            # 因为在某个时刻解码时，如果use_cache为True，就会保存每一层计算得到的key_states和value_states
            # 即保存在下面的 next_decoder_cache 参数中。所以这里需要分别要对应取每一层对应的 key_states 和value_states
            # past_key_values[i][j] 标记第i层的key_states
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None)
            else:
                layer_outputs = decoder_layer(hidden_states,
                                              attention_mask=attention_mask,
                                              position_ids=position_ids,  # 这里组要注意， 每一个Layer都要输入位置编码
                                              past_key_value=past_key_value,
                                              output_attentions=output_attentions,  # 默认 False
                                              use_cache=use_cache)

            hidden_states = layer_outputs[0]
            # 取解每个码层最后的输出，形状为 [batch_size, seq_len, hidden_size]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
                # if output_attentions: layer_outputs = (hidden_states, self_attn_weights, present_key_value)
                # else: layer_outputs = (hidden_states, present_key_value)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        # 多层解码器最后一层的输出，形状为 [batch_size, seq_len, hidden_size]

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        # 将本次解码缓存得到的key_states 和 value_states 传入给下一个时刻

        if not return_dict:  # 模型return_dict=True，即不以如下方式进行返回
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                       past_key_values=next_cache,
                                       hidden_states=all_hidden_states,
                                       attentions=all_self_attns)  # 默认以该结构返回


class NormHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.first_flag = True

    def forward(self, hidden_states):
        if self.training:
            norm_weight = nn.functional.normalize(self.weight)
            self.first_flag = True
        elif self.first_flag:
            self.first_flag = False
            self.weight.data = nn.functional.normalize(self.weight)
            norm_weight = self.weight
        else:
            norm_weight = self.weight
        return nn.functional.linear(hidden_states, norm_weight)


_init_weights = True


@contextmanager
def no_init_weights(_enable=True):
    global _init_weights
    old_init_weights = _init_weights
    if _enable:
        _init_weights = False
    try:
        yield
    finally:
        _init_weights = old_init_weights


class BaichuanForCausalLM(BaichuanPreTrainedModel):
    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config, *model_args, **model_kwargs)
        self.model = BaichuanModel(config)

        self.lm_head = NormHead(config.hidden_size, config.vocab_size, bias=False)
        if hasattr(config, "quantization_config") and isinstance(config.quantization_config,
                                                                 dict) and config.quantization_config.get(
            'load_in_4bit', False):
            try:
                from .quantizer import quantize_offline, init_model_weight_int4
            except ImportError:
                raise ImportError(f"Needs QLinear to run quantize.")
            quantize_offline(self, 4)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
            cache_dir: Optional[Union[str, os.PathLike]] = None,
            ignore_mismatched_sizes: bool = False,
            force_download: bool = False,
            local_files_only: bool = False,
            token: Optional[Union[str, bool]] = None,
            revision: str = "main",
            use_safetensors: bool = None,
            **kwargs,
    ):
        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            # 如果 config 不是一个 PretrainedConfig 类对象，默认为None,即会从本地config.json文件中读取
            config_path = config if config is not None else pretrained_model_name_or_path
            # cls.config_class.from_pretrained() 调用 BaichuanConfig() 得到一个配置类对象
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=False,
                proxies=None,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder="",
                _from_auto=False,
                _from_pipeline=None,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        if hasattr(config, "quantization_config") and config.quantization_config['load_in_4bit']:
            # 是否进行量化，默认不进行。 所谓量化是指降低模型的参数精度，例如由bf16变为4bit，这样可以降低模型的计算复杂度和大小
            # 但代价是损失一部分的模型精度。
            # bf16（bfloat16）是16位浮点数格式，其中有1位符号位、8位指数位和7位尾数位。
            # fp16（float16）也是16位浮点数格式，但通常采用1位符号位、5位指数位和10位尾数位。
            try:
                from .quantizer import init_model_weight_int4
                from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map
                from accelerate.utils import CustomDtype
                from accelerate.utils import get_balanced_memory
            except ImportError:
                raise ImportError(f"Needs import model weight init func to run quantize.")
                # Instantiate model.
            init_contexts = [no_init_weights(_enable=True)]
            init_contexts.append(init_empty_weights())
            with ContextManagers(init_contexts):
                model = cls(config)

            model_file = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')
            state_dict = torch.load(model_file, map_location="cpu")
            model.is_quantized = True

            device_map = kwargs.pop("device_map", None)
            torch_dtype = kwargs.pop("torch_dtype", None)

            if device_map is not None:
                kwargs = {"no_split_module_classes": model._no_split_modules}
                target_dtype = CustomDtype.INT4
                max_memory = get_balanced_memory(
                    model,
                    dtype=target_dtype,
                    low_zero=(device_map == "balanced_low_0"),
                    max_memory=None,
                    **kwargs)
                kwargs["max_memory"] = max_memory
                device_map = infer_auto_device_map(model, dtype=target_dtype, **kwargs)

            model = init_model_weight_int4(config, model, state_dict)

            # Set model in evaluation mode to deactivate DropOut modules by default
            model.eval()
            # If it is a model with generation capabilities, attempt to load the generation config
            if model.can_generate():
                try:
                    model.generation_config = GenerationConfig.from_pretrained(
                        pretrained_model_name_or_path,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=False,
                        proxies=None,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder="",
                        _from_auto=False,
                        _from_pipeline=None,
                        **kwargs,
                    )
                except (OSError, TypeError):
                    logger.info(
                        "Generation config file not found, using a generation config created from the model config."
                    )
                    pass

            if device_map is not None:
                dispatch_model(model, device_map=device_map)

            return model
        return super(BaichuanForCausalLM, cls).from_pretrained(pretrained_model_name_or_path, *model_args,
                                                               config=config, cache_dir=cache_dir,
                                                               ignore_mismatched_sizes=ignore_mismatched_sizes,
                                                               force_download=force_download,
                                                               local_files_only=local_files_only, token=token,
                                                               revision=revision,
                                                               use_safetensors=use_safetensors, **kwargs)
        # 返回根据预训练模型实例化的BaichuanForCausalLM类对象

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 返回结果是否输出注意力 True or False，默认情况下为False
        # 这里的self.config是上面super(BaichuanForCausalLM, cls).from_pretrained()传入的config对象,它一部分成员变量来自于本地的config.json文件
        # 一部分是config类继承自，BaichuanConfig和 Transformers中的PretrainedConfig类，
        # 后这便包含有 output_attentions, output_hidden_states 等相关默认参数
        # 如果需要输出可在config.json中添加一行 "output_attentions": true
        # 不过事实上本代码也不支持返回注意力权重矩阵，因为memory_efficient_attention和
        # caled_dot_product_attention两个函数的返回结果都不包含注意力权重矩阵。
        print("i am in BaichuanForCausalLM begin, input shape", input_ids.shape)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        # 同 output_attentions 含义类似，如需返回可在config.json中添加一行 "output_hidden_states": true
        print("验证1 BaichuanForCausalLM中 return_dict 的取值=======", return_dict)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 以字典形式返回结果，此处返回的结果为Fasel
        print("验证2 BaichuanForCausalLM中 return_dict 的取值=======", return_dict)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,  # [batch_size, seq_len]
            attention_mask=attention_mask,  # 默认为None即可
            position_ids=position_ids,  # 默认为None即可
            past_key_values=past_key_values,  # 默认为None即可
            inputs_embeds=inputs_embeds,  # 为 None即可
            use_cache=use_cache,  # 默认为 True
            output_attentions=output_attentions,  # 默认为 False
            output_hidden_states=output_hidden_states,  # 默认为 False
            return_dict=return_dict,  # 默认为 False
        )

        hidden_states = outputs[0]  # BaichuanModel模块输出的hidden_state [batch_size, seq_len, hidden_size]
        print("验证，use_cache = True 时，BaichuanForCausalLM中 hidden_states，预期输出形状为[1,1,4096]",hidden_states.shape)
        logits = self.lm_head(hidden_states)  # 分类结果 [batch_size, seq_len, vocab_size]
        print("验证，use_cache = True 时， BaichuanForCausalLM中 logits，预期输出形状为[1,vocab_size]", logits.shape)
        loss = None
        if labels is not None: # 训练时，计算损失
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous() # [batch_size, seq_len-1, vocab_size]
            shift_labels = labels[..., 1:].contiguous() # [batch_size, seq_len-1]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size) # [batch_size*seq_len-1, vocab_size]
            shift_labels = shift_labels.view(-1)# [batch_size*seq_len-1]
            softmax_normalizer = shift_logits.max(-1).values ** 2
            z_loss = self.config.z_loss_weight * softmax_normalizer.mean()
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels) + z_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # print("i am in BaichuanForCausalLM end ",outputs.past_key_values[0][0].shape)
        print("=======------------==========")
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        这个函数是用来在模型推理时构造输入，在 transformers/generation/utils.py sample()方法 中会调用这个函数，关键几行代码如下：
        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self(**model_inputs,return_dict=True,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            ...
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)


        :param input_ids: 这里传入的input_ids是整个完整cat后的结果
        :param past_key_values:
        :param attention_mask:
        :param inputs_embeds:
        :param kwargs:
        :return:
        """
        print(f"每次前向传播前输入模型的参数use_cache = {past_key_values != None} ：{kwargs}")
        if past_key_values: # 编码是时为None，后续如果use_cache = True，则不为 None
            print("prepare_inputs_for_generation",input_ids.shape)
            print("prepare_inputs_for_generation",input_ids)
            input_ids = input_ids[:, -1:] # use_cache = Ture 时只取最后一个token，否则全部输入模型解码下一个时刻
            print("prepare_inputs_for_generation", input_ids)

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def quantize(self, bits: int):
        try:
            from .quantizer import quantize_online
        except ImportError:
            raise ImportError(f"Needs QLinear to run quantize.")
        return quantize_online(self, bits)

    def chat(self, tokenizer, messages: List[dict], stream=False,
             generation_config: Optional[GenerationConfig] = None):
        print("i am in 883")
        # 在使用chat()之前会通过以下方式来得到配置类
        # model.generation_config = GenerationConfig.from_pretrained("./Baichuan2_7B_Chat")
        generation_config = generation_config or self.generation_config
        print("i am in 885")
        print(f"原始message: {messages}")
        input_ids = build_chat_input(self, tokenizer, messages, generation_config.max_new_tokens)
        print("build_chat_input处理messages后的input_ids: ", input_ids)
        if stream:
            streamer = TextIterStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            Thread(target=self.generate, kwargs=dict(
                inputs=input_ids, streamer=streamer,
                generation_config=generation_config,
            )).start()
            return streamer
        else:
            outputs = self.generate(input_ids, generation_config=generation_config)
            response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True) # 解码成汉字
            return response
