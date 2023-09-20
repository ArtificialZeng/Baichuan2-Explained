# Copyright (c) 2023, Baichuan Intelligent Technology. All rights reserved.

from .configuration_baichuan import BaichuanConfig           # 导入当前包下的 `configuration_baichuan` 模块中的 `BaichuanConfig` 类。
from .generation_utils import build_chat_input, TextIterStreamer  # 导入当前包下的 `generation_utils` 模块中的 `build_chat_input` 和 `TextIterStreamer`。

import math  # 导入Python的内置数学函数库。
from threading import Thread  # 导入Python的多线程库中的 `Thread` 类。
from typing import List, Optional, Tuple, Union  # 导入Python的类型注释库，这里导入了 `List`、`Optional`、`Tuple` 和 `Union`。

import torch  # 导入PyTorch框架。
from torch import nn  # 从PyTorch中导入神经网络库。
from torch.nn import CrossEntropyLoss  # 从PyTorch的神经网络库中导入交叉熵损失函数。
from torch.nn import functional as F  # 从PyTorch的神经网络库中导入功能模块，并为其取别名`F`。
from transformers import PreTrainedModel, PretrainedConfig  # 从`transformers`库中导入 `PreTrainedModel` 和 `PretrainedConfig`。
from transformers.activations import ACT2FN  # 从`transformers`库中导入激活函数的映射表 `ACT2FN`。
from transformers.generation.utils import GenerationConfig  # 从`transformers`库中的`generation.utils`模块导入`GenerationConfig`。
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast  # 从`transformers`库中的`modeling_outputs`模块导入`BaseModelOutputWithPast`和`CausalLMOutputWithPast`。
from transformers.utils import logging, ContextManagers  # 从`transformers`库中的`utils`模块导入`logging`和`ContextManagers`。

import os  # 导入Python的内置OS模块，用于处理操作系统相关任务。
from contextlib import contextmanager  # 从Python的`contextlib`模块导入`contextmanager`，它用于创建上下文管理器。
from accelerate import init_empty_weights  # 从`accelerate`库中导入`init_empty_weights`函数。

logger = logging.get_logger(__name__)  # 使用`transformers`提供的日志功能创建一个日志器对象，`__name__`是当前模块的名称。

# 试图从 `xformers` 库中导入 `ops` 模块，并为其取别名`xops`。
try:
    from xformers import ops as xops
except ImportError:  # 如果导入失败（即没有正确安装`xformers`库）
    xops = None
    logger.warning(
        "Xformers is not installed correctly. If you want to use memory_efficient_attention to accelerate training use the following command to install Xformers\npip install xformers."
    )

# 定义了一个辅助函数`_get_interleave`。
def _get_interleave(n):
    # 内嵌函数`_get_interleave_power_of_2`，用于计算并返回一个列表。
    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    # 根据`n`是否是2的整数次幂来调用内部函数，并返回一个列表。
    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            _get_interleave_power_of_2(closest_power_of_2)
            + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )

# 定义了一个辅助函数`_fill_with_neg_inf`。
def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

# 定义了一个辅助函数`_buffered_future_mask`。
def _buffered_future_mask(tensor, maxpos, alibi, attn_heads):
    _future_mask = torch.triu(_fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1)
    _future_mask = _future_mask.unsqueeze(0) + alibi
    new_future_mask = _future_mask.to(tensor)
    return new_future_mask[: tensor.shape[0] * attn_heads, :maxpos, :maxpos]

# 定义了一个辅助函数`_gen_alibi_mask`。
def _gen_alibi_mask(tensor, n_head, max_pos):
    slopes = torch.Tensor(_get_interleave(n_head))
    position_point = torch.arange(max_pos) - max_pos + 1
    position_point = position_point.unsqueeze(0).unsqueeze(0).expand(n_head, -1, -1)
    diag = torch.diag(position_point[0])
    position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(_fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1)
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask

# 定义了一个层归一化的变体：RMSNorm。
class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, epsilon=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size))
        self.epsilon = epsilon

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states

# 定义了一个多层感知机(MLP)类。
class MLP(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias


# 定义 BaichuanAttention 类，是一个 PyTorch 神经网络模块
class BaichuanAttention(torch.nn.Module):
    
    # 构造函数，接收 BaichuanConfig 配置类实例
    def __init__(self, config: BaichuanConfig):
        super().__init__()  # 调用父类的构造函数
        self.config = config  # 保存传入的配置
        self.hidden_size = config.hidden_size  # 从配置中取出隐藏层大小
        self.num_heads = config.num_attention_heads  # 从配置中取出注意力头数
        self.head_dim = self.hidden_size // self.num_heads  # 计算每个注意力头的维度
        self.max_position_embeddings = config.model_max_length  # 从配置中取出模型的最大长度
        
        # 确保 hidden_size 可以被 num_heads 整除
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )
        
        # 定义一个线性层，用于获取查询、键和值
        self.W_pack = torch.nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=False
        )
        
        # 定义一个输出线性层
        self.o_proj = torch.nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
    
    # 辅助函数，用于重新整形张量以适应注意力计算
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()  # 获取输入的批次大小、序列长度和隐藏层大小

        # 使用线性变换获取查询、键和值
        proj = self.W_pack(hidden_states)
        proj = (
            proj.unflatten(-1, (3, self.hidden_size))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
        )
        query_states = (
            proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key_states = (
            proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )

        # 如果给定了过去的键值对，则连接它们
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # 根据 use_cache 的值来决定是否保存键和值
        past_key_value = (key_states, value_states) if use_cache else None

        # 判断是否有 xformers，并根据条件使用不同的注意力计算方式
        if xops is not None and self.training:
            attn_weights = None
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask)
            attn_output = attn_output.transpose(1, 2)
        else:
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

            # 对注意力权重应用掩码
            if attention_mask is not None:
                if q_len == 1:  # 缓存中的推断
                    if len(attention_mask.size()) == 4:
                        attention_mask = attention_mask[:, :, -1:, :]
                    else:
                        attention_mask = attention_mask[:, -1:, :]
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )
            
            # 获取注意力权重并计算注意力输出
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)  # 对最后一个维度进行 softmax，得到注意力权重
            attn_output = torch.matmul(attn_weights, value_states)  # 使用权重对 value 进行加权求和得到注意力输出

            attn_output = attn_output.transpose(1, 2)  # 调换维度，以便之后的处理

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)  # 重新整形，以得到最终输出的形状
        attn_output = self.o_proj(attn_output)  # 通过输出的线性层

        # 如果不输出注意力权重，则设置为 None
        if not output_attentions:
            attn_weights = None

        # 返回注意力输出，注意力权重（如果需要）和过去的键值对（如果使用缓存）
        return attn_output, attn_weights, past_key_value

            

class BaichuanLayer(torch.nn.Module):  # 定义一个名为 "BaichuanLayer" 的 PyTorch 模型类，它继承了torch.nn.Module

    def __init__(self, config: BaichuanConfig):  # 构造函数接收一个名为config的BaichuanConfig类型参数
        super().__init__()  # 调用父类(torch.nn.Module)的初始化方法
        self.hidden_size = config.hidden_size  # 从config中提取hidden_size并赋值给类变量self.hidden_size
        self.self_attn = BaichuanAttention(config=config)  # 用config初始化BaichuanAttention对象，并赋值给self.self_attn
        self.mlp = MLP(  # 初始化一个MLP对象，并赋值给self.mlp
            hidden_size=self.hidden_size,  # MLP的hidden_size参数等于我们之前设置的self.hidden_size
            intermediate_size=config.intermediate_size,  # 从config中提取intermediate_size作为MLP的参数
            hidden_act=config.hidden_act,  # 从config中提取hidden_act作为MLP的参数
        )
        self.input_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)  # 初始化一个RMSNorm对象，并赋值给self.input_layernorm
        self.post_attention_layernorm = RMSNorm(  # 初始化另一个RMSNorm对象，并赋值给self.post_attention_layernorm
            config.hidden_size, epsilon=config.rms_norm_eps
        )

    def forward(  # 定义前向传播函数
        self,
        hidden_states: torch.Tensor,  # 输入参数为一个tensor，代表隐藏状态
        attention_mask: Optional[torch.Tensor] = None,  # 可选的attention_mask参数，默认为None
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 可选的past_key_value参数，默认为None
        output_attentions: Optional[bool] = False,  # 可选的output_attentions参数，默认为False
        use_cache: Optional[bool] = False,  # 可选的use_cache参数，默认为False
    ) -> Tuple[  # 函数的返回类型为一个Tuple
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states  # 将输入的hidden_states保存为residual以便后面使用

        hidden_states = self.input_layernorm(hidden_states)  # 将hidden_states通过self.input_layernorm进行处理

        # Self Attention部分
        hidden_states, self_attn_weights, present_key_value = self.self_attn(  # 使用self.self_attn处理hidden_states，并返回三个结果
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states  # 将处理后的hidden_states与原始的residual进行加和，实现residual connection

        # Fully Connected部分
        residual = hidden_states  # 更新residual为处理后的hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)  # 将hidden_states通过self.post_attention_layernorm进行处理
        hidden_states = self.mlp(hidden_states)  # 将hidden_states通过self.mlp进行处理
        hidden_states = residual + hidden_states  # 再次使用residual connection

        outputs = (hidden_states,)  # 将处理后的hidden_states放入outputs tuple中

        if use_cache:  # 如果use_cache为True
            outputs += (present_key_value,)  # 将present_key_value也加入到outputs tuple中

        return outputs  # 返回outputs tuple



class BaichuanPreTrainedModel(PreTrainedModel):  # 定义一个名为 "BaichuanPreTrainedModel" 的类，继承了 "PreTrainedModel" 类
    config_class = BaichuanConfig  # 定义一个类属性，将BaichuanConfig赋值给config_class，用于指定该模型的配置类是什么

    base_model_prefix = "model"  # 定义一个类属性，其值是字符串 "model"，通常用于标识模型的主要子模块或基础模型

    supports_gradient_checkpointing = True  # 定义一个类属性，标识该模型支持梯度检查点功能，可以节省显存但会使计算稍慢

    _no_split_modules = ["BaichuanLayer"]  # 定义一个类属性，包含模型内不应该被拆分为子模型的模块名称

    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]  # 定义一个类属性，列出在加载模型时应该忽略的意外键名

    def _init_weights(self, module):  # 定义一个方法用于初始化模型权重
        std = self.config.initializer_range  # 从模型的配置中获取权重初始化的标准差
        
        if isinstance(module, torch.nn.Linear):  # 如果传入的模块是线性层
            module.weight.data.normal_(mean=0.0, std=std)  # 使用正态分布初始化线性层的权重，均值为0，标准差为std
            if module.bias is not None:  # 如果线性层有偏置
                module.bias.data.zero_()  # 使用0来初始化偏置

        elif isinstance(module, torch.nn.Embedding):  # 如果传入的模块是嵌入层
            module.weight.data.normal_(mean=0.0, std=std)  # 使用正态分布初始化嵌入层的权重，均值为0，标准差为std
            if module.padding_idx is not None:  # 如果嵌入层有填充索引
                module.weight.data[module.padding_idx].zero_()  # 将填充索引对应的嵌入向量初始化为0

    def _set_gradient_checkpointing(self, module, value=False):  # 定义一个方法用于设置模块的梯度检查点
        if isinstance(module, BaichuanModel):  # 如果传入的模块是BaichuanModel类型
            module.gradient_checkpointing = value  # 设置模块的梯度检查点属性为给定的value值



class BaichuanModel(BaichuanPreTrainedModel):  # 定义一个名为 "BaichuanModel" 的类，它继承了 "BaichuanPreTrainedModel" 类

    def __init__(self, config: BaichuanConfig):  # 定义类的构造函数，接受一个名为 "config" 的参数，该参数是一个BaichuanConfig对象
        super().__init__(config)  # 调用父类 (BaichuanPreTrainedModel) 的构造函数并传入config参数
        self.padding_idx = config.pad_token_id  # 从config中获取pad_token_id并设置为类的属性
        self.vocab_size = config.vocab_size  # 从config中获取词汇表的大小并设置为类的属性
        self.n_head = config.num_attention_heads  # 从config中获取注意力头的数量并设置为类的属性
        self.embed_tokens = torch.nn.Embedding(  # 创建一个嵌入层
            config.vocab_size, config.hidden_size, self.padding_idx  # 指定词汇表大小、嵌入大小和填充索引
        )
        self.layers = torch.nn.ModuleList(  # 创建一个模块列表
            [BaichuanLayer(config) for _ in range(config.num_hidden_layers)]  # 根据隐藏层的数量多次实例化BaichuanLayer模块
        )
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)  # 创建一个RMSNorm正则化层

        self.gradient_checkpointing = config.gradient_checkpointing  # 设置梯度检查点属性
        self.post_init()  # 调用post_init方法，可能在父类中定义，用于进一步的初始化
        self.max_cache_pos = config.model_max_length  # 设置模型的最大缓存位置
        self.first_run = True  # 设置一个标志，表示模型是否是第一次运行
        self.alibi_mask = None  # 初始化一个alibi_mask属性，值为None

    def get_input_embeddings(self):  # 定义一个方法用于获取输入的嵌入
        return self.embed_tokens  # 返回嵌入层

    def set_input_embeddings(self, value):  # 定义一个方法用于设置输入的嵌入
        self.embed_tokens = value  # 将传入的嵌入赋值给嵌入层属性

    def get_alibi_mask(self, tensor, seq_length_with_past):  # 定义一个方法用于获取alibi mask
        if self.training:  # 如果模型处于训练模式
            slopes = torch.Tensor(_get_interleave(self.n_head))  # 获取交错值
            position_point = (
                torch.arange(seq_length_with_past) - seq_length_with_past + 1  # 计算位置点
            )
            position_point = (
                position_point.unsqueeze(0)
                .unsqueeze(0)
                .expand(self.n_head, seq_length_with_past, -1)  # 调整位置点的形状并扩展
            )
            diag = torch.diag(position_point[0])  # 获取对角线
            position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(
                -1, -2
            )  # 调整位置点的形状
            alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point  # 计算alibi值
            mask = _buffered_future_mask(  # 获取未来的mask
                tensor, seq_length_with_past, alibi, self.n_head
            )
        else:  # 如果模型处于评估模式
            if self.first_run:  # 如果是第一次运行
                self.first_run = False  # 设置标志为False
                self.register_buffer(  # 注册一个缓冲区
                    "future_mask",
                    _gen_alibi_mask(tensor, self.n_head, self.max_cache_pos).to(
                        tensor
                    ),
                    persistent=False,  # 使缓冲区不持久
                )
            if seq_length_with_past > self.max_cache_pos:  # 如果当前序列长度超过最大缓存位置
                self.max_cache_pos = seq_length_with_past  # 更新最大缓存位置
                self.register_buffer(  # 再次注册一个缓冲区
                    "future_mask",
                    _gen_alibi_mask(tensor, self.n_head, self.max_cache_pos).to(
                        tensor
                    ),
                    persistent=False,  # 使缓冲区不持久
                )
            mask = self.future_mask[
                : self.n_head, :seq_length_with_past, :seq_length_with_past
            ]  # 获取未来的mask
        return mask  # 返回mask


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot provide both input_ids and inputs_embeds simultaneously"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You need to provide input_ids or inputs_embeds")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        seq_length_with_past = seq_length

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.training:
            if (
                self.alibi_mask is None
                or self.alibi_mask.shape[-1] != seq_length_with_past
            ):
                self.alibi_mask = self.get_alibi_mask(
                    inputs_embeds, seq_length_with_past
                )
            alibi_mask = self.alibi_mask
        else:
            alibi_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)

        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                expanded_mask = attention_mask.to(alibi_mask.dtype)
                expanded_mask = torch.tril(
                    torch.gt(expanded_mask[:, :, None] * expanded_mask[:, None, :], 0)
                ) * torch.eq(expanded_mask[:, :, None] - expanded_mask[:, None, :], 0)
            else:
                expanded_mask = attention_mask
            bsz = inputs_embeds.size(0)
            src_len, tgt_len = alibi_mask.size()[-2:]
            expanded_mask = (
                expanded_mask.unsqueeze(1)
                .expand(bsz, 1, src_len, tgt_len)
                .to(alibi_mask.dtype)
            )
            inverted_mask = 1.0 - expanded_mask
            inverted_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(alibi_mask.dtype).min
            )
            attention_mask = inverted_mask + alibi_mask.unsqueeze(0)
        else:
            attention_mask = alibi_mask

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

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
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


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
        #if hasattr(config, "quantization_config") and config.quantization_config['load_in_4bit']:
        if hasattr(config, "quantization_config") and isinstance(config.quantization_config, dict) and config.quantization_config.get('load_in_4bit', False):
            try:
                from .quantizer import quantize_offline, init_model_weight_int4
            except ImportError:
                raise ImportError(f"Needs quantize_offline to run quantize.")
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
            config_path = config if config is not None else pretrained_model_name_or_path
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
                    **kwargs,
                )
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
                config=config, cache_dir=cache_dir, ignore_mismatched_sizes=ignore_mismatched_sizes, 
                force_download=force_download, local_files_only=local_files_only, token=token, revision=revision, 
                use_safetensors=use_safetensors, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            softmax_normalizer = shift_logits.max(-1).values ** 2
            z_loss = self.config.z_loss_weight * softmax_normalizer.mean()
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels) + z_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def quantize(self, bits: int):
        try:
            from .quantizer import quantize_online
        except ImportError:
            raise ImportError(f"Needs QLinear to run quantize.")
        return quantize_online(self, bits)
        
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past_key_values
        )

    def _build_chat_input(
        self, tokenizer, messages: List[dict], max_new_tokens: int = 0
    ):
        max_new_tokens = max_new_tokens or self.generation_config.max_new_tokens
        max_input_tokens = self.config.model_max_length - max_new_tokens
        max_input_tokens = max(self.config.model_max_length // 2, max_input_tokens)
        total_input, round_input = [], []
        for i, message in enumerate(messages[::-1]):
            content_tokens = tokenizer.encode(message["content"])
            if message["role"] == "user":
                round_input = (
                    [self.generation_config.user_token_id]
                    + content_tokens
                    + round_input
                )
                if (
                    total_input
                    and len(total_input) + len(round_input) > max_input_tokens
                ):
                    break
                else:
                    total_input = round_input + total_input
                    if len(total_input) >= max_input_tokens:
                        break
                    else:
                        round_input = []
            elif message["role"] == "assistant":
                round_input = (
                    [self.generation_config.assistant_token_id]
                    + content_tokens
                    + [self.generation_config.eos_token_id]
                    + round_input
                )
            else:
                raise ValueError(f"message role not supported yet: {message['role']}")
        total_input = total_input[-max_input_tokens:]  # truncate left
        total_input.append(self.generation_config.assistant_token_id)
        total_input = torch.LongTensor([total_input]).to(self.device)
        return total_input

    def chat(self, tokenizer, messages: List[dict], stream=False,
             generation_config: Optional[GenerationConfig]=None):
        generation_config = generation_config or self.generation_config
        input_ids = build_chat_input(self, tokenizer, messages, generation_config.max_new_tokens)
        if stream:
            streamer = TextIterStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            Thread(target=self.generate, kwargs=dict(
                inputs=input_ids, streamer=streamer,
                generation_config=generation_config,
            )).start()
            return streamer
        else:
            outputs = self.generate(input_ids, generation_config=generation_config)
            response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            return response