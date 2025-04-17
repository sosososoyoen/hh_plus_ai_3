# 기본 과제 : 데이터 전처리에 따른 학습 모델 평가 비교

## 토크나이저를 활용하여 전제와 가설 문장들을 연결해서 학습시켰을 경우

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") # 전처리 함수 정의
def def preprocess_function(examples): 
	return tokenizer( examples["premise"], examples["hypothesis"], 
	truncation=True, padding="longest", )
```
- BERT 계열의 토크나이저는 문장쌍까지만 구조적으로 구분 가능 => premise, hypothesis 문장 따로 전달

```
tokenizer(
    text = "문장1",              # 또는
    text_pair = "문장2",         # 문장쌍일 때
    truncation = True,           # 너무 길면 자름
    padding = "longest",         # 가장 긴 입력의 길이에 맞춰서 패딩 토큰을 자동으로 채워줌
    max_length = 128,            # 최대 길이
    return_tensors = "pt"        # torch 텐서로 반환 (선택)
)
```

- `[CLS] premise [SEP] hypothesis [SEP]` 구조를 사용하기로 함
- token_type_ids(문장을 구분해주는 인덱스)을 통해 모델이 문장을 쉽게 분리할 수 있도록 처리

[7365/7365 21:08, Epoch 3/3]

| Epoch | Training Loss | Validation Loss | Accuracy | F1       |
| ----- | ------------- | --------------- | -------- | -------- |
| 1     | 0.866200      | 0.868188        | 0.603328 | 0.594561 |
| 2     | 0.796400      | 0.844906        | 0.624222 | 0.624192 |
| 3     | 0.732300      | 0.867255        | 0.623738 | 0.623072 |

Out[54]:

TrainOutput(global_step=7365, training_loss=0.7982978111210115, metrics={'train_runtime': 1268.5338, 'train_samples_per_second': 742.97, 'train_steps_per_second': 5.806, 'total_flos': 130220993318850.0, 'train_loss': 0.7982978111210115, 'epoch': 3.0})


## 스페셜 토큰 없이 전제와 가설을 이어서 한 문장으로 만들어서 학습 시켰을 경우

```python
def preprocess_function(data):
    text = []
    for i in range(len(data['premise'])):
        text.append(data['premise'][i] + data['hypothesis'][i])
    return tokenizer(text, truncation=True)
```

| Epoch | Training Loss | Validation Loss | Accuracy | F1       |
| ----- | ------------- | --------------- | -------- | -------- |
| 1     | 0.932600      | 0.958541        | 0.527903 | 0.530360 |
| 2     | 0.899600      | 0.951936        | 0.532397 | 0.533510 |
| 3     | 0.852900      | 0.969038        | 0.533938 | 0.534887 |

TrainOutput(global_step=7365, training_loss=0.8950229787211473, metrics={'train_runtime': 354.5992, 'train_samples_per_second': 2657.882, 'train_steps_per_second': 20.77, 'total_flos': 39329901033570.0, 'train_loss': 0.8950229787211473, 'epoch': 3.0}


## 모델 평가 비교
![output (1)](https://github.com/user-attachments/assets/904cb7df-6a48-4dea-9e71-223a21b15414)
