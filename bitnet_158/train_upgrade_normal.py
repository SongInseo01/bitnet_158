import torch
from normal_Transformer import Transformer
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import os

# 최적화된 학습 설정
batch_size = 16  # 배치 크기 증가 (GPU 메모리 허용 범위 내에서)
block_size = 2048  # 시퀀스 길이 단축으로 메모리 사용량 감소
max_iteration = 10000
eval_interval = 1  # 평가 간격 증가
learning_rate = 1e-3  # 학습률 조정
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iteration = 200
n_embed = 32
n_head = 4
n_layer = 4
dropout = 0.1

# 멀티 GPU 설정
use_multi_gpu = torch.cuda.device_count() > 1
print(f"Available GPUs: {torch.cuda.device_count()}")

# 데이터셋 로드 및 캐싱
dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")
train_data = dataset["train"]

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 최적화된 전처리 함수
def preprocess(example):
    input_text = example["input"]
    output_text = example["output"]
    full_text = f"### Question: {input_text}\n### Answer: {output_text}"

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=block_size,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = tokenized["input_ids"].squeeze()
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone()  # clone() 사용으로 메모리 효율성 향상
    }

# 데이터셋 전처리 적용 (캐싱 사용)
processed_dataset = train_data.map(
    preprocess, 
    remove_columns=train_data.column_names,
    num_proc=4,  # 병렬 처리
    load_from_cache_file=True  # 캐시 사용
)
processed_dataset.set_format("torch")

# DataLoader 구성 (멀티 워커 사용)
train_loader = DataLoader(
    processed_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=4,  # 멀티 워커로 데이터 로딩 속도 향상
    pin_memory=True,  # GPU 전송 속도 향상
    persistent_workers=True  # 워커 재사용
)

# 모델 초기화
model = Transformer(vocab_length=tokenizer.vocab_size).to(device)

# 멀티 GPU 설정
if use_multi_gpu:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

# 최적화된 옵티마이저 (AdamW with weight decay)
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=learning_rate,
    weight_decay=0.01,  # 정규화
    betas=(0.9, 0.95)   # 더 안정적인 베타 값
)

# 학습률 스케줄러 추가
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=max_iteration,
    eta_min=learning_rate * 0.1
)

# Mixed Precision 학습을 위한 GradScaler
scaler = GradScaler()

# 그래디언트 누적 설정
accumulation_steps = 1  # BitNet은 효율적이므로 누적 없이도 충분

# 학습 루프
model.train()
accumulated_loss = 0.0

for step, batch in enumerate(train_loader):
    if step >= max_iteration:
        break

    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)

    # Mixed Precision 학습
    with autocast():
        # BitNet 모델은 (logits, loss) 튜플을 반환
        logits, loss = model(input_ids, labels)
        
        # 멀티 GPU 환경에서 loss 처리
        if use_multi_gpu and isinstance(loss, torch.Tensor):
            loss = loss.mean()  # DataParallel에서 각 GPU의 loss를 평균
        
        # loss가 스칼라인지 확인
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()
        
        loss = loss / accumulation_steps  # 누적 스텝만큼 나누기

    # 그래디언트 스케일링 및 역전파
    scaler.scale(loss).backward()
    accumulated_loss += loss.item()

    # 그래디언트 누적
    if (step + 1) % accumulation_steps == 0:
        # 그래디언트 클리핑
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 옵티마이저 스텝
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    # 평가 및 로그
    if step % eval_interval == 0:
        avg_loss = accumulated_loss / eval_interval if step > 0 else loss.item()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Step {step} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        accumulated_loss = 0.0

# 모델 저장
save_path = "./normal_transformer_optimized.pth"
if use_multi_gpu:
    torch.save(model.module.state_dict(), save_path)  # DataParallel 사용 시
else:
    torch.save(model.state_dict(), save_path)
print(f"모델이 저장되었습니다: {save_path}")

# 메모리 정리
torch.cuda.empty_cache()
print("학습 완료 및 메모리 정리됨")