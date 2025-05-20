# [8주차] 기본과제 - LoRA rank에 따른 학습 성능 비교해보기

- [X]  `lora_r`를 `[8, 128, 256]`로 변화시켜가며 학습
    - Deepspeed 없이 순수 LoRA만을 가지고 기존과 같은 LLM(`facebook/opt-350m`)과 dataset(`sahil2801/CodeAlpaca-20k`)를 활용합니다.
    - Rank를 8, 128, 256로 바꿔가며 학습을 진행해봅니다.
     
- [X] Rank에 따른 loss, 학습 속도, 그리고 메모리 점유율 공유

- [X] LoRA의 장단점 분석

### 학습 환경

**Colab Enterprise**

머신 유형: g2-standard-4

GPU 유형: NVIDIA_L4 x 1

지역: asia-northeast3

시스템 RAM : 15.6 GB

GPU RAM : 22.5 GB



### **모델**

`facebook/opt-350m`

- https://huggingface.co/facebook/opt-350m

### **데이터셋**

`sahil2801/CodeAlpaca-20k` 

- 빠른 성능 비교를 위해 split 50%만 사용
- https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k

### LoRa config

```
task_type=TaskType.CAUSAL_LM,
inference_mode=False,
r=r_value,
lora_dropout=0.1,
lora_alpha=r_value * 4,
target_modules=["q_proj", "k_proj", "v_proj"]
```

- r_value = [8, 128, 256]
- lora_alpha 값은 r * 4으로 랭크의 값과 비례하게 조정
- 타겟 모듈 : 어텐션 메커니즘의 q, k, v 레이어

### train/loss

![W B Chart 2025  5  20  오후 5_46_26](https://github.com/user-attachments/assets/a4857f9c-6bf9-468b-86d2-990e60c0776e)


https://api.wandb.ai/links/soyeon-bubbles/y5q8y7ko

1. **안정권으로 들어가는 시점**
    - r=256: 2회차 만에 1.5 이하로 진입 → 빠르게 안정권으로
    - r=128: 3회차에 1.5 이하 → 중간
    - r=8: 5회차가 돼야 겨우 1.75 정도 → 느림
2. **학습 속도(Runtime)**
    - r=8 ⇒ 761.3346
    - r=128 ⇒ 822.9918
    - r=256 ⇒ 907.3956
3. **최종 손실 값**
    - r=256: 1.33 (최저)
    - r=128: 1.43 (r=256 대비 +0.10)
    - r=8: 1.72 (r=128 대비 +0.29)
4. **용량 대비 효과**
    - r 값 8→128로 키우면 손실 감소폭이 크지만, 128→256 구간은 효과가 점점 줄어든다. (=기울기 감소)

### 결론

- **성능 우선 ⇒** r=256이 가장 좋음. 빠른 안정화.
- **효율적인 선** ⇒ r=128
- **리소스가 한정적이고, 최소한의 학습만** ⇒ r=8 (그러나 손실이 크다)

### GPU Memory Allocated

![W B Chart 2025  5  20  오후 11_53_59](https://github.com/user-attachments/assets/d2e21829-4590-4feb-9e34-6b529ab292c7)


https://api.wandb.ai/links/soyeon-bubbles/lkaxivz4

| rank (r) | 평균 GPU Memory Allocated (%) |
| --- | --- |
| **256** | 약 **20.6%** |
| **128** | 약 **19.9%** |
| **8** | 약 **18.8%** |
- 차이가 1-2%p 정도로 미미함

### **Process Memory In Use (MB)**

![W B Chart 2025  5  20  오후 5_47_53](https://github.com/user-attachments/assets/5639499e-9b22-4f16-aa20-23ae4eaeb8c0)


https://api.wandb.ai/links/soyeon-bubbles/0qfyanlj

| rank r | 평균 메모리 (MB) | 표준편차 (MB) | 최소 (MB) | 최대 (MB) |
| --- | --- | --- | --- | --- |
| **8** | 1898 | 58.7 | 1874 | 2402 |
| **128** | 1917 | 17.4 | 1885 | 1932 |
| **256** | 1964 | 47.0 | 1879 | 2026 |
1. **메모리 사용량 증가**
    - r 값이 커질수록 메모리 사용량도 증가함
    - r=8 → 약 1898 MB,
        
        r=128 → 약 1917 MB (+19 MB),
        
        r=256 → 약 1964 MB (+66 MB) 정도 차이.
        
2. **초기 스파이크 & 변동성**
    - r=8은 첫 측정 시 2400 MB까지 찍고 곧바로 1880 MB 선으로 내려감 → 모델 로딩, 캐시 구축 등에 따른 일시적인 현상으로 추정
    - 이후 r=8은 크게 변하지 않고, r=128은 ±20 MB 내에서, r=256은 ±50 MB 내에서 비교적 주기적인 메모리 할당/해제가 반복됨
3. **안정성 관점**
    - r=128 ⇒ 표준편차가 가장 작음
    - r=8 ⇒ 일회성 스파이크를 제외하면 낮은 메모리를 사용함.
    - r=256은 당연히 가장 많은 메모리 사용함. 큰 랭크 연산을 지속해야 하므로 메모리 풀링/해제가 반복되며 변동 폭은 중간

### 결론

- **메모리 절약이 최우선 ⇒** r=8
- **메모리 안정성이 중요 ⇒** r=128
- **성능 최우선 ⇒** r=256

### **GPU Time Spent Accessing Memory (%)**

![W B Chart 2025  5  20  오후 5_59_52](https://github.com/user-attachments/assets/3711c255-3c6d-4304-9529-808c38e4f279)


https://api.wandb.ai/links/soyeon-bubbles/grpmpy33

1. **메모리 대역폭 의존도**
    - r=256 > r=128 > r=8 순으로 메모리 대역폭 의존도가 높아짐
2. **안정성**
    - r=128이 가장 일관된 메모리 접근 패턴을 보임
3. **병목 파악**
    - r=256 쪽 병목이 더 두드러짐

### 결론

- 연산/메모리 효율이 가장 안정적 →  r=128
