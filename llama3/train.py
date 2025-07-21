import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import time
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from model import ModelArgs, Transformer

# 모델 및 토크나이저 설정
model_args = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    vocab_size=32000,  # Llama 3 토크나이저에 맞게 설정
    max_seq_len=2048,
    max_batch_size=8
)

# 데이터셋 로드
dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")
train_data = dataset["train"]

# 사용자 정의 데이터셋 클래스
class MedicalFlashcardsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return tokens.squeeze(0)

# 토크나이저 초기화 (Llama 3 토크나이저 사용)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer.pad_token = tokenizer.eos_token

# 데이터셋 및 데이터로더 생성
train_dataset = MedicalFlashcardsDataset(train_data, tokenizer, model_args.max_seq_len)
train_loader = DataLoader(
    train_dataset,
    batch_size=model_args.max_batch_size,
    shuffle=True
)

# 모델 초기화
model = Transformer(model_args).cuda()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for data parallelism")
    model = torch.nn.DataParallel(model)

# 옵티마이저 및 손실 함수 설정
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# 학습 파라미터
num_epochs = 3
accumulation_steps = 4  # 그라디언트 누적
log_interval = 50

# 학습 루프
def train():
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            tokens = batch.cuda()
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:].contiguous()
            
            # Forward pass
            logits = model(inputs, start_pos=0)
            
            # 손실 계산
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            
            # 그라디언트 누적
            loss = loss / accumulation_steps
            loss.backward()
            
            # 가중치 업데이트
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
            # 로깅
            if batch_idx % log_interval == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                current_loss = total_loss / log_interval
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {current_loss:.4f} | Time: {elapsed:.2f}s")
                total_loss = 0
                start_time = time.time()
        
        # 에포크 종료 시 체크포인트 저장
        checkpoint = {
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, f"llama3_medical_checkpoint_epoch{epoch+1}.pt")

# 학습 시작
if __name__ == "__main__":
    train()