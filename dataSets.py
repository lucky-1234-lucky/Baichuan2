from typing import List, Dict, Tuple, Any
import torch
from loguru import logger
import transformers
from transformers import AutoTokenizer
from torch.utils.data import Dataset

import com


def format(instance: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, Any]] = []
    for i, msg in enumerate(instance["conversations"]):
        if i == 0:
            messages.append({"role": "system", "content": msg["system"]})
        messages.extend(
            [
                {"role": "user", "content": msg["query"]},
                {"role": "assistant", "content": msg["answer"]},
            ]
        )
    return messages


def preprocessing(
    tokenizer: AutoTokenizer,
    instance: Dict[str, Any],
    model_max_length: int = 1024,
    user_tokens: List[int] = [195],
    assistant_tokens: List[int] = [196],
    ignore_index: List[int] = [-100],
):
    def _parse_messages(
        messages: List[Dict[str, str]], split_role="user"
    ) -> Tuple[str, List[List[Dict[str, str]]]]:
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            # 结束一轮对话才将数据添加到 rounds
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:  # 最后的数据也添加到 rounds
            rounds.append(round)

        return system, rounds

    system, messages = _parse_messages(format(instance))
    system_ids = tokenizer.encode(system)
    input_ids, labels = [], []
    # 多轮或单轮对话, 即[query, answer, query, answer, ...] 或 [query, answer]
    for i, rounds in enumerate(messages[::-1]):
        tmp_input, tmp_label = [], []
        for round in rounds:  # 一次对话, 即 [query, answer]
            role, content = round["role"], round["content"]
            value_ids = tokenizer.encode(content)
            if role == "user":
                tmp_input.extend(user_tokens + value_ids)
                tmp_label.extend(
                    [tokenizer.eos_token_id] + ignore_index * len(value_ids)
                )
            else:
                tmp_input.extend(assistant_tokens + value_ids)
                tmp_label.extend(ignore_index + value_ids)

        if i == 0:  # 最后一次对话需要在末尾添加</s>
            tmp_input.append(tokenizer.eos_token_id)
            tmp_label.append(tokenizer.eos_token_id)

        # 如果对话时 token 数量比 model_max_length 多, 则丢弃前几轮对话
        if len(system_ids + tmp_input + input_ids) > model_max_length:
            break
        else:
            input_ids = tmp_input + input_ids
            labels = tmp_label + labels
    # 添加 system
    input_ids: List[int] = system_ids + input_ids
    labels: List[int] = ignore_index * len(system_ids) + labels

    # 填充 pad
    input_ids.extend((model_max_length - len(input_ids)) * [tokenizer.pad_token_id])
    labels.extend((model_max_length - len(labels)) * ignore_index)

    # 转 Tensor
    input_ids = torch.LongTensor(input_ids)
    labels = torch.LongTensor(labels)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


class SupervisedDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        model_max_length: int,
        user_tokens: List[int] = [195],
        assistant_tokens: List[int] = [196],
        ignore_index: List[int] = [-100],
    ):
        """
        Args:
            data_path: 文件或文件夹路径
            tokenizer:
            model_max_length: 模型最大长度
            user_tokens:
            assistant_tokens:
            ignore_index:
        """
        super().__init__()
        self.dataset: List[Dict[str, Any]] = com.read_datas(data_path)
        self.tokenizer: AutoTokenizer = tokenizer
        self.model_max_length: int = model_max_length
        self.user_tokens: List[List] = user_tokens
        self.assistant_tokens: List[int] = assistant_tokens
        self.ignore_index: List[int] = ignore_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        data_dic: Dict[str, torch.Tensor] = preprocessing(
            self.tokenizer,
            self.dataset[idx],
            self.model_max_length,
            self.user_tokens,
            self.assistant_tokens,
            self.ignore_index,
        )
        if idx == 0:
            input_ids: List[int] = data_dic["input_ids"].tolist()
            text = self.tokenizer.decode(input_ids)
            label_ids: List[int] = data_dic["labels"].tolist()
            label = self.tokenizer.decode(
                [label for label in label_ids if label != -100]
            )
            logger.info(f"user_ids: {len(input_ids)}  {input_ids}")
            logger.debug(f"user: {text}")
            logger.info(f"label_ids: {len(label_ids)}  {label_ids}")
            logger.debug(f"label: {label}")

        return data_dic


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/Baichuan2-13B-Chat",
        use_fast=False,
        trust_remote_code=True,
        cache_dir=None,
    )

    supervisedDataset = SupervisedDataset(
        data_path="/root/autodl-tmp/Baichuan2-our/datas/handled/devs/battle_death",
        tokenizer=tokenizer,
        model_max_length=2048,
    )
    for data in supervisedDataset:
        pass
