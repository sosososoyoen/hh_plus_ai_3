
## 📘 개요

본 문서는 LLM이 생성한 논문 요약 결과에 대해 ROUGE와 BERTScore를 기준으로 평가한 결과를 정리한 문서입니다. (with ChatGPT 4o)

사용 모델 : gpt-4o-mini
정답용 모델 : gpt-4o

---

## 🧠 종합 평가 요약


- 의미 전달: ****우수함****

- 표현 유사성: ****보통****

- 전반적인 요약 품질: ****BERTScore 기준 사람 수준 요약에 근접****


---

  

## 📊 점수 상세

![다운로드 (4)](https://github.com/user-attachments/assets/89148e51-4997-48d6-bfdc-2b5f9304406d)


### 🔷 ROUGE


| 지표        | 점수   | 설명 |
|-------------|--------|------|
| ROUGE-1     | 0.2543 | 단어 수준 유사도 |
| ROUGE-2     | 0.0702 | 2-gram 연속성 유사도 (문장 구성 구조) |
| ROUGE-L     | 0.2197 | 문장 뼈대의 유사성 |
| ROUGE-Lsum  | 0.2428 | 전체 요약 구조 유사성 |

  

### 🔷 BERTScore

  

| 지표        | 점수   | 설명 |
|-------------|--------|------|
| Precision   | 0.7095 | 정답 요약 의미 중 포함된 비율 |
| Recall      | 0.6887 | 내가 쓴 요약이 정답을 얼마나 커버했는지 |
| F1          | 0.6990 | 의미 유사성 종합 점수 (사람 기준으로 매우 양호함) |

  

---

  

## 📝 평가 의견

  

- 의미적으로는 상당히 좋은 요약이며, 핵심 내용을 잘 전달함

- 다만 문장 구성 방식, 표현 스타일은 논문 요약 스타일과 차이가 있음

- 표현 구조를 더 논문스러운 문장으로 다듬을 경우 ROUGE 점수 상승 기대

  

---

  

## ✅ 개선 제안

  

1. ****프롬프트 수정****  

   → "논문 스타일로 학술적인 요약을 해줘" 같은 지시 문구 사용

  

2. ****예시 문장 추가****  

   → 학습 시 논문 스타일 예시 문장을 함께 제시

  

3. ****문단 단위 요약 후 통합****  

   → Recall + Precision 균형 잡기
  
