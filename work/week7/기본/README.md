# [7주차] 기본과제: Validation data를 포함하여 Fine-tuning 해보기

## 태스크
GPT fine-tuning을 할 때 validation data를 두어 validation loss도 같이 측정하는 코드 구현하기

## 요구사항

- [x] Validation data 준비
- [x] 학습 시 validation loss 계산
    - Trainer를 정의할 때 validation data를 추가하고 validation data에 대한 evaluation을 진행하도록 수정합니다. 
    - 실제로 학습 후, `train/loss`와 `eval/loss` 에 해당하는 wandb log를 공유해주시면 됩니다.


## 해결

### test.py 수정

1. validation 데이터를 가져온다.
```python
train_dataset = lm_datasets["train"]

eval_dataset = lm_datasets["validation"]
```
2. Trainer의 인자에 `eval_dataset`을 지정해준다.
```python
trainer = Trainer(

model=model,

args=training_args,

train_dataset=train_dataset,

eval_dataset=eval_dataset,

tokenizer=tokenizer,

data_collator=default_data_collator

)
```

**학습 파라미터**
```bash
!python ./train.py \

--model_name_or_path openai-community/openai-gpt \

--per_device_train_batch_size 8 \

--dataset_name wikitext \

--dataset_config_name wikitext-2-raw-v1 \

--do_train \

--output_dir /tmp/test-clm \

--save_total_limit 1 \

--logging_steps 100 \

--do_eval \

--eval_strategy steps \

--load_best_model_at_end \

--report_to wandb
```


**eval/loss**

https://wandb.ai/soyeon-bubbles/Hanghae99/reports/eval-loss-25-05-15-17-41-08---VmlldzoxMjc4NjM3OQ

![W B Chart 2025  5  15  오후 1_06_46](https://github.com/user-attachments/assets/440a3c04-153e-44a7-9510-593a0b3bc5cb)


**train/loss**

https://wandb.ai/soyeon-bubbles/Hanghae99/reports/train-loss-25-05-15-17-41-38---VmlldzoxMjc4NjM5MQ

![W B Chart 2025  5  15  오후 1_06_56](https://github.com/user-attachments/assets/13a55358-da21-4355-afff-2f32c9190a7e)
