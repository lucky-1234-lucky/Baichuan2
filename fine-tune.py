from typing import Tuple
from peft import LoraConfig, TaskType, get_peft_model
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from argument import DataArguments, ModelArguments, TrainingArguments, LoraArguments
from dataSets import SupervisedDataset


def init_model(
    model_args: ModelArguments, training_args: TrainingArguments
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )

    return tokenizer, model


def train():
    # 初始化参数
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = (
        parser.parse_args_into_dataclasses()
    )

    # 初始化模型
    tokenizer, model = init_model(model_args, training_args)
    # 是否使用 lora
    if lora_args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["W_pack"],
            inference_mode=False,
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    trainset = SupervisedDataset(
        data_args.data_path, tokenizer, training_args.model_max_length
    )
    # 训练模型
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=trainset, tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
