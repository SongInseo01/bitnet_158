import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from BitLoRA.peft import BitLoraConfig, get_peft_model

# 1. 모델 및 토크나이저 로드
model_id = "tiiuae/Falcon-E-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, revision="prequantized")
tokenizer.pad_token = tokenizer.eos_token  # SFTTrainer에서 pad token 명시

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    revision="prequantized"
)

# 3. BitLoRA 구성
bitlora_config = BitLoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    use_dora=False
)

model = get_peft_model(model, bitlora_config)

# 4. 예시 데이터셋 불러오기
dataset = load_dataset("Abirate/english_quotes")
dataset = dataset["train"].train_test_split(test_size=0.1)

# 5. 전처리 함수 정의
def preprocess_function(example):
    return tokenizer(example["quote"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 6. 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./bitlora-falcon3b-sft",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    learning_rate=2e-4,
    weight_decay=0.01,
    bf16=True,
    remove_unused_columns=False,
    report_to="none",
    push_to_hub=False
)

# 7. SFTTrainer 정의
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    max_seq_length=128,
    dataset_text_field="quote",
    packing=False  # 문장 여러 개 합치지 않고 단일 문장 학습
)

# 8. 학습 수행
trainer.train()

# 9. 필요시 저장
trainer.save_model("./bitlora-falcon3b-sft-final")
tokenizer.save_pretrained("./bitlora-falcon3b-sft-final")
