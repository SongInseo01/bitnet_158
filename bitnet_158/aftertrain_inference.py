import torch
from Bitnet_Transformer import BitnetTransformer
from transformers import AutoTokenizer
import torch.nn.functional as F

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 하이퍼파라미터는 저장 시와 일치하게 맞춰야 함
n_embed = 32
n_head = 4
n_layer = 4
dropout = 0.1
block_size = 256

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.pad_token = tokenizer.eos_token  # 필요 시 eos_token으로 설정

# 모델 생성 후 파라미터 로드
model = BitnetTransformer(vocab_length=tokenizer.vocab_size).to(device)
model.load_state_dict(torch.load("bitnet_transformer.pth", map_location=device))
model.eval()

# 추론 함수
def generate_text(prompt, max_new_tokens=256):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

# 예시 추론
prompt = "### Question: What is the main symptom of diabetes?\n### Answer:"
output = generate_text(prompt, max_new_tokens=256)
print("=== 생성된 응답 ===")
print(output)
