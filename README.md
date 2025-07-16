# BitNet 1.58 기반 경량 Transformer 모델 (PyTorch)

이 프로젝트는 BitNet 1.58 양자화 기법을 적용한 경량 Transformer 언어 모델입니다.
`Multi-Head Attention`, `FeedForward`, `Block`, `Positional Encoding`, `Token Embedding` 등을 모두 BitLinear 기반으로 구성하여 경량화와 학습 효율성을 극대화합니다.

---

## 🧠 핵심 기능

- BitNet 1.58 방식의 3값 양자화 (`{-1, 0, 1}`) 기반 `BitLinear` 모듈
- 8bit 정밀도 입력 양자화 + RMSNorm 정규화
- 멀티헤드 어텐션과 FFN 포함된 GPT-style Block
- 학습 / 추론 코드 포함
- HuggingFace 데이터셋 및 토크나이저 사용

---

## 📂 프로젝트 구조

```
bitnet_158/
├── BitLinear.py               # BitNet 1.58 양자화 구현
├── Bitnet_Transformer.py      # 전체 Transformer 모델 정의
├── train.py                   # 학습 루프
├── aftertrain_inference.py    # 추론 스크립트
```

---

## ▶️ 학습 실행

```bash
python train.py
```

학습 완료 후 `bitnet_transformer.pth`로 모델이 저장됩니다.

---

### 💬 추론 실행

```bash
python aftertrain_inference.py
```

출력 예시:

```
=== 생성된 응답 ===
### Question: What is the main symptom of diabetes?
### Answer: Increased thirst, frequent urination, fatigue...
```

---

## 📦 사용한 데이터셋

- [`medalpaca/medical_meadow_medical_flashcards`](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)

---

## 🧪 주요 기술 스택

- PyTorch
- HuggingFace Datasets / Tokenizers
- BitNet 1.58 Quantization
- SimpleRMSNorm
- Causal Language Modeling

---
