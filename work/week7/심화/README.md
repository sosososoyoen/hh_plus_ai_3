# [7주차] 심화과제: 자신만의 text 파일로 LLM Instruction-tuning 해보기

## 태스크
자체 제작한 data를 활용하여 LLM instruction-tuning 해보기

## 요구사항

- [ ]  Instruction-data 준비
    - 먼저 text corpus를 `corpus.json`의 이름으로 준비합니다.
    - Corpus의 형식은 제한이 없고, 100개 이상의 sample들로 구성되어 있으면 됩니다.
- [ ]  Train 및 validation data 준비
    - 먼저 `corpus.json`를 불러옵니다.
    - 그 다음 8:2 비율로 나눠, train과 validation data를 나눕니다.
    - 그 다음 기존의 data 전처리 코드를 적절히 수정하여 불러온 train과 validation data를 전처리합니다.

## 결과

**학습 파라미터**
```
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=512,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    logging_steps=5,
    eval_steps=5,
    do_train=True,
    do_eval=True,
    num_train_epochs=5,
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="wandb",
)
```


**eval/loss**

https://api.wandb.ai/links/soyeon-bubbles/q31d8h78


![W B Chart 2025  5  16  오전 2_56_42](https://github.com/user-attachments/assets/d2a79eac-b4ef-4164-8d61-43c2cba1887a)


**train/loss**

[https://api.wandb.ai/links/soyeon-bubbles/aectozv7](https://api.wandb.ai/links/soyeon-bubbles/aectozv7)

![W B Chart 2025  5  16  오전 2_56_52](https://github.com/user-attachments/assets/3cb8068a-bb84-4045-a513-d3d26ca57019)

