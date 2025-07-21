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


from typing import Optional
import torch
import torch.nn as nn
from .bitnet import BitLinear

def replace_linear_with_bitnet_linear(model, previous_dtype: Optional[torch.dtype] = None):
    """
    """
    # Recursively replace linear layers
    if previous_dtype is None:
        previous_dtype = torch.get_default_dtype()

        model_dtype = model.dtype
        torch.set_default_dtype(model_dtype)

        previous_dtype = model_dtype

    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear_with_bitnet_linear(module, previous_dtype=previous_dtype)
        
        # Replace nn.Linear layers, but skip 'lm_head'
        if name != 'lm_head' and isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None

            with torch.device(module.weight.device):
                # Create a new instance of the custom linear layer
                new_layer = BitLinear(in_features, out_features, bias=bias)
                # Copy weights and biases
                with torch.no_grad():
                    new_layer.weight.copy_(module.weight)
                    if bias:
                        new_layer.bias.copy_(module.bias)
            
            # Replace the layer in the model
            setattr(model, name, new_layer)
    return model

# 2. BitLinear로 Linear 대체
model = replace_linear_with_bitnet_linear(model)

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
