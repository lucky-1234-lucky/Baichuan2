from typing import Optional
from dataclasses import dataclass, field
import transformers


@dataclass
class DataArguments:
    data_path: str = field(
        default="/root/autodl-tmp/Baichuan2-our/datas/handled/data1/trains",
        metadata={"help": "Path to the training data."},
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="/root/autodl-tmp/Baichuan2-13B-Chat"
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    report_to: str = field(default="none")
    output_dir: str = field(default="/root/autodl-tmp/Baichuan2-our/output/data1")
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=2)
    save_strategy: str = field(default="epoch")
    learning_rate: float = field(default=2e-5)
    lr_scheduler_type: str = field(default="constant")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.98)
    adam_epsilon: float = field(default=1e-8)
    max_grad_norm: float = field(default=1.0)
    weight_decay: float = field(default=1e-4)
    warmup_ratio: float = field(default=0.0)
    logging_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)
    deepspeed: str = field(default="/root/autodl-tmp/Baichuan2-our/ds_config.json")
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)


@dataclass
class LoraArguments:
    use_lora: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.1)
