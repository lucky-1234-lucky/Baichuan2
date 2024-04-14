from typing import Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from peft import PeftModel
from loguru import logger
import com


def init_model(
    base_model_path: str,
    lora_model_path: Optional[str] = None,
):
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, use_fast=False, trust_remote_code=True
    )
    # 加载 base LLM
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

    # 加载 LoRA 模型
    if lora_model_path:
        model = PeftModel.from_pretrained(
            model,
            lora_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    return tokenizer, model


if __name__ == "__main__":
    data_path = "/root/autodl-tmp/Baichuan2-our/datas/handled/data3/devs"
    model_name_or_path = "/root/autodl-tmp/merge"
    lora_model_path = None

    tokenizer, model = init_model(model_name_or_path, lora_model_path)
    for path in com.get_paths(data_path):
        convers: Dict[str, Any] = com.load(path)["conversations"][0]
        messages = [
            {"role": "system", "content": convers["system"]},
            {"role": "user", "content": convers["query"]},
            {"role": "assistant", "content": convers["answer"]},
        ]
        response = model.chat(tokenizer, messages[:-1])
        logger.info(messages[-1]["content"])
        logger.debug(response)
        com.dump(
            f"{path}".replace("datas/handled", "results1111"),
            {"conversations": messages + [{"role": "result", "content": response}]},
        )
