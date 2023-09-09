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


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="baichuan-inc/Baichuan2-7B-Base")


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
        user_tokens=[195],
        assistant_tokens=[196],
    ):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        self.ignore_index = -100
        item = self.preprocessing(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
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
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
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
    )
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
