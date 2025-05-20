import os
import torch
import pandas as pd
import argparse
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

"""
pip install peft transformers wandb datasets scikit-learn pandas ipython trl
"""


def setup_wandb(r_value):
    wandb.init(project='Hanghae99')
    wandb.run.name = 'LoRA_basic_' + str(r_value)


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
    return tokenizer, model


def format_prompts(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts


def setup_lora(model, r_value):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r_value,
        lora_dropout=0.1,
        lora_alpha=r_value * 4,
        target_modules=["q_proj", "k_proj", "v_proj"]
    )
    return get_peft_model(model, peft_config)


def main():
    # wandb 세팅
    setup_wandb(args.r)

    # 모델, 토크나이저 불러오기
    tokenizer, model = load_model_and_tokenizer()

    # 데이터셋 불러오기
    train_ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train[:50%]")

    # 데이터 collaltor 세팅
    # 프롬프트 쪽은 loss 무시하고 답변(completion)만 loss 계산
    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # LoRA 세팅
    model = setup_lora(model, args.r)

    # Initialize trainer
    trainer = SFTTrainer(
        model,
        train_dataset=train_ds,
        args=SFTConfig(output_dir="./clm-instruction-tuning", max_seq_length=128),
        formatting_func=format_prompts,
        data_collator=collator,
    )

    # 트레이닝 GoGo
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=int, default=8, help='LoRA rank parameter')
    args = parser.parse_args()
    main()
