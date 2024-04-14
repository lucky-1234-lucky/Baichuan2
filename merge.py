import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from loguru import logger


def merge_lora_to_base_model(
    model_name_or_path: str, adapter_name_or_path: str, save_path: str
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # 合并权重
    logger.info("merging weight...")
    model = PeftModel.from_pretrained(model, adapter_name_or_path)
    model = model.merge_and_unload()
    logger.info("weight is merged")

    # 保存 tokenizer 和 model
    logger.info("saving weight...")
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    logger.info("weight is saved")


if __name__ == "__main__":
    model_name_or_path = "/root/autodl-tmp/Baichuan2-13B-Chat"
    adapter_name_or_path = "/root/autodl-tmp/Baichuan2-our/output/data3/checkpoint-734"
    save_path = "/root/autodl-tmp/merge"
    merge_lora_to_base_model(model_name_or_path, adapter_name_or_path, save_path)
