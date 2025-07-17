import torch
from Bitnet_Transformer import BitnetTransformer
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# 학습 값
batch_size = 4
block_size = 4096
max_iteration = 10000
eval_interval = 1
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iteration = 200
n_embed = 32
n_head = 4
n_layer = 4
dropout = 0.1

# 데이터셋 로드
dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")
train_data = dataset["train"]

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # 또는 직접 학습한 tokenizer
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 전처리
def preprocess(example):
    input_text = example["input"]
    output_text = example["output"]
    full_text = f"### Question: {input_text}\n### Answer: {output_text}"

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=block_size,
        padding="max_length"
    )

    input_ids = tokenized["input_ids"]
    return {
        "input_ids": input_ids,
        "labels": input_ids.copy()
    }


# 데이터셋 전처리 적용
processed_dataset = train_data.map(preprocess, remove_columns=train_data.column_names)
processed_dataset.set_format("torch")

# DataLoader 구성
train_loader = DataLoader(processed_dataset, batch_size=batch_size, shuffle=True)

# 학습 루프
model = BitnetTransformer(vocab_length=tokenizer.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step, batch in enumerate(train_loader):
    if step >= max_iteration:
        break

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    logits, loss = model(input_ids, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

# 모델 저장
save_path = "./bitnet_transformer.pth"
torch.save(model.state_dict(), save_path)
print(f"모델이 저장되었습니다: {save_path}")
