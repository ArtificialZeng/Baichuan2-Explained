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


# @dataclass 是一个Python装饰器，用于自动生成初始化、比较等特殊方法。它使得数据类定义变得简洁。
@dataclass
# 这行定义了一个名为 `ModelArguments` 的类。
class ModelArguments:
    # 定义了一个可选的字符串类型的类属性 `model_name_or_path`，其默认值为 `"baichuan-inc/Baichuan2-7B-Base"`。
    model_name_or_path: Optional[str] = field(default="baichuan-inc/Baichuan2-7B-Base")

# 同第一行，用于自动生成特殊方法。
@dataclass
# 定义了一个名为 `DataArguments` 的类。
class DataArguments:
    # 定义了一个字符串类型的类属性 `data_path`，其默认值为 `None`。还为该字段添加了一些元数据描述。
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )

# 同上，用于自动生成特殊方法。
@dataclass
# 定义了一个名为 `TrainingArguments` 的类，该类继承自 `transformers.TrainingArguments`。
class TrainingArguments(transformers.TrainingArguments):
    # 定义了一个可选的字符串类型的类属性 `cache_dir`，其默认值为 `None`。
    cache_dir: Optional[str] = field(default=None)
    # 定义了一个字符串类型的类属性 `optim`，其默认值为 `"adamw_torch"`。
    optim: str = field(default="adamw_torch")
    # 定义了一个整型属性 `model_max_length`，其默认值为 `512`，并为它提供了描述。
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # 定义了一个布尔值属性 `use_lora`，默认为 `False`。
    use_lora: bool = field(default=False)

# 定义了一个名为 `SupervisedDataset` 的类，继承自 `Dataset` 类。
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    # 定义了 `SupervisedDataset` 类的初始化方法，并接收一系列参数。
    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length,
        user_tokens=[195],
        assistant_tokens=[196],
    ):
        # 调用父类（Dataset）的初始化方法。
        super(SupervisedDataset, self).__init__()
        # 读取由 `data_path` 参数指定的JSON文件，并将其内容赋值给 `self.data`。
        self.data = json.load(open(data_path))
        # 这几行将传入的参数赋值给相应的类属性。
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        self.ignore_index = -100
        # 对第一个数据进行预处理，并打印它的输入。
        item = self.preprocessing(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        # 对预处理后的标签进行解码，并打印解码后的内容。
        for id_ in item["labels"]:
            if id_ == -100:
                continue

            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    # 定义魔法方法，返回数据集的大小。
    def __len__(self):
        return len(self.data)

    # 定义预处理方法，对单个示例进行预处理。
    def preprocessing(self, example):
        # 初始化输入ID的空列表。
        input_ids = []
        # 初始化标签的空列表。
        labels = []

        # 遍历每个对话中的消息。
        for message in example["conversations"]:
            # 获取消息的发送者。
            from_ = message["from"]
            # 获取消息的内容。
            value = message["value"]
            # 使用tokenizer对消息内容进行编码。
            value_ids = self.tokenizer.encode(value)

            # 如果消息来自人类用户。
            if from_ == "human":
                # 在输入ID中添加用户特定的token和消息token。
                input_ids += self.user_tokens + value_ids
                # 在标签中添加结束符和忽略标签。
                labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(
                    value_ids
                )
            else:
                # 如果消息来自助手，添加助手特定的token和消息token。
                input_ids += self.assistant_tokens + value_ids
                # 在标签中添加忽略标签和消息token。
                labels += [self.ignore_index] + value_ids

        # 在输入ID和标签的末尾都追加结束符token。
        input_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)

        # 对输入ID和标签进行截断。
        input_ids = input_ids[: self.model_max_length]
        labels = labels[: self.model_max_length]

        # 为输入ID和标签填充token。
        input_ids += [self.tokenizer.pad_token_id] * (
            self.model_max_length - len(input_ids)
        )
        labels += [self.ignore_index] * (self.model_max_length - len(labels))

        # 转换输入ID和标签为PyTorch的LongTensor。
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)

        # 创建注意力掩码。
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # 返回预处理后的结果。
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    # 定义魔法方法，允许使用索引从数据集中获取示例。
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


def train():
    # 使用transformers库的HfArgumentParser创建一个参数解析器
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    
    # 解析命令行参数并将它们分别分配给model_args, data_args, 和 training_args
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 加载预训练的因果语言模型（Causal Language Model）
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,  # 从model_args获取模型名称或路径
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    
    # 加载预训练的分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,  # 同样地，从model_args获取模型名称或路径
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    
    # 检查training_args是否有use_lora标志设置为True
    if training_args.use_lora:
        # 如果上述条件为真，导入与LoRA (Layer-wise Recomputation) 相关的模块
        from peft import LoraConfig, TaskType, get_peft_model

        # 配置LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["W_pack"],
            inference_mode=False,
            r=1,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        
        # 在模型中启用输入的梯度计算
        model.enable_input_require_grads()
        
        # 获取与LoRA相关的模型
        model = get_peft_model(model, peft_config)
        
        # 打印模型中的可训练参数
        model.print_trainable_parameters()

    # 创建一个有监督的数据集
    dataset = SupervisedDataset(
        data_args.data_path, tokenizer, training_args.model_max_length
    )
    
    # 创建一个训练器
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    
    # 开始训练模型
    trainer.train()
    
    # 保存训练器的状态
    trainer.save_state()
    
    # 将训练好的模型保存到指定的目录
    trainer.save_model(output_dir=training_args.output_dir)

# Python的标准模式，确保代码作为主程序运行时才执行下面的内容
if __name__ == "__main__":
    # 调用上面定义的train函数，开始训练过程
    train()

