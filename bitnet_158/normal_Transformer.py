# 멀티헤드 어텐션 만들기 + 피드포워드 + 블록스

import torch
import torch.nn as nn
import torch.nn.functional as F
from zeta.nn.modules.simple_rmsnorm import SimpleRMSNorm

batch_size = 16
block_size = 2048
max_iteration = 10000
eval_interval = 1
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iteration = 200
n_embed = 32
n_head = 4
n_layer = 4
dropout = 0.1

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, inputs):
        batch_size, sequence_length, embedding_dim = inputs.shape
        keys = self.key(inputs)
        queries = self.query(inputs)
        weights = queries @ keys.transpose(-2, -1) * (embedding_dim ** -0.5)
        weights = weights.masked_fill(self.tril[:sequence_length, :sequence_length] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        values = self.value(inputs)
        output = weights @ values
        return output
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, inputs):
        return torch.cat([head(inputs) for head in self.heads], dim=-1)
    

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.SiLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, input_tensor):
        return self.layer(input_tensor)
    

class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.attention = MultiHeadAttention(n_heads, head_size)
        self.feed_dorward = FeedForward(n_embed)
        self.layer_norm1 = SimpleRMSNorm(n_embed)
        self.layer_norm2 = SimpleRMSNorm(n_embed)

    def forward(self, input_tensor):
        input_tensor = input_tensor + self.attention(self.layer_norm1(input_tensor))
        input_tensor = input_tensor + self.feed_dorward(self.layer_norm2(input_tensor))
        return input_tensor


class Transformer(nn.Module):
    def __init__(self, vocab_length):
        super().__init__()
        self.embedding_token_table = nn.Embedding(vocab_length, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, 4) for _ in range(n_layer)])
        self.ln_f = SimpleRMSNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_length)

    def forward(self, inputs, targets=None):
        batch, sequence = inputs.shape

        token_embed = self.embedding_token_table(inputs)
        pos_embed = self.position_embedding_table(torch.arange(sequence, device=device))
        x = token_embed + pos_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch, sequence, embed_size = logits.shape
            logits = logits.view(batch * sequence, embed_size)
            targets = targets.view(batch * sequence)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, inputs, max_new_tokens):
        for _ in range(max_new_tokens):
            inputs_cond = inputs[:, -block_size:]

            logits, loss = self(inputs_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_inputs = torch.multinomial(probs, num_samples=1)
            inputs = torch.cat((inputs, next_inputs), dim=1)
        return inputs

