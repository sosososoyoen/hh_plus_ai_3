import os
import json
import random
import logging
import wandb
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator
)
import datasets
import transformers
import sys
from transformers.trainer_utils import get_last_checkpoint

MODEL_NAME_OR_PATH = "openai-community/openai-gpt"
CORPUS_PATH = "./corpus.json"
OUTPUT_DIR = "./gpt_finetuned"
BLOCK_SIZE = 512
BATCH_SIZE = 1
ACCUMULATION_STEPS = 4     
LOGGING_STEPS  = 5 
SAVE_TOTAL_LIMIT = 1 


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64,garbage_collection_threshold:0.6")
wandb.init(project='Hanghae99')
wandb.run.name = 'gpt_instruction-tuning-2'

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
random.shuffle(data)
split_idx = int(0.8 * len(data))
train_list = data[:split_idx]
valid_list = data[split_idx:]

ds = DatasetDict({
    "train": Dataset.from_list(train_list),
    "validation": Dataset.from_list(valid_list),
})


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)


special_tokens = {"eos_token": "", "pad_token": "<|pad|>"}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))


BLOCK_SIZE = min(BLOCK_SIZE, tokenizer.model_max_length)

# 전처리
def preprocess_fn(examples):
    texts = []
    for inp, out in zip(examples.get("input", []), examples.get("output", [])):
        if inp is None or out is None:
            continue
        texts.append(inp + tokenizer.eos_token + out)
    if not texts:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    model_inputs = tokenizer(
        texts,
        truncation=True,
        max_length=BLOCK_SIZE,
        padding="max_length",
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# None 값이 있는 예시 제거
for split in ["train", "validation"]:
    ds[split] = ds[split].filter(
        lambda ex: ex.get("input") is not None and ex.get("output") is not None
    )


tokenized = ds.map(
    preprocess_fn,
    batched=True,
    remove_columns=["input", "output"],
)


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=ACCUMULATION_STEPS,
    eval_strategy="steps",
    logging_steps=LOGGING_STEPS,
    eval_steps=LOGGING_STEPS,
    do_train=True,
    do_eval=True,
    num_train_epochs=5,
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=True,
    report_to="wandb",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

# 로거 설정
logger = logging.getLogger()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if training_args.should_log:
    transformers.utils.logging.set_verbosity_info()  # log level을 INFO로 변경 

log_level = training_args.get_process_log_level()

# 우리가 가지고 있는 logger와 HuggingFace의 logger의 log level 설정
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

# 기타 HuggingFace logger option들을 설정
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

logger.info(f"Training/evaluation parameters {training_args}")
checkpoint = None
last_checkpoint = get_last_checkpoint(training_args.output_dir)  # 만약 output_dir에 checkpoint가 남아있으면 이를 사용하고, 없으면 None이 return됩니다.
if training_args.resume_from_checkpoint is not None:  # output_dir이 아닌 다른 위치에서의 checkpoint를 resume_from_checkpoint로 지정할 수 있습니다.
    checkpoint = training_args.resume_from_checkpoint
else:  # 아니면 last_checkpoint로 checkpoint를 지정합니다.  
    checkpoint = last_checkpoint
    
train_result = trainer.train(resume_from_checkpoint=None)

trainer.save_model()

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()