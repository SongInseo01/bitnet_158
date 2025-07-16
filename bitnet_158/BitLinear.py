import torch
from torch import nn, Tensor
import torch.nn.functional as F
from zeta.nn.modules.simple_rmsnorm import SimpleRMSNorm


def activation_quant(x: Tensor) -> Tensor:
    """
    8비트 정밀도로 활성화 양자화를 수행 (토큰 단위, 그룹 없음)
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    return (x * scale).round().clamp_(-128, 127) / scale


def weight_quant(w: Tensor, eps: float = 1e-5) -> Tensor:
    """
    BitNet 방식의 1.58비트 양자화 수행:
    - 평균 절댓값으로 정규화 후
    - 반올림 및 [-1, 1] 범위로 클리핑
    결과는 {-1, 0, +1} 값만 가짐
    """
    gamma = w.abs().mean()
    w_scaled = w / (gamma + eps)
    return torch.round(w_scaled).clamp_(-1, 1)


class BitLinear(nn.Module):
    """
    BitNet 1.58 방식 양자화 선형 계층.
    입력은 RMS 정규화 + 8비트 양자화,
    가중치는 1.58bit (3값) 양자화.
    """
    def __init__(self, in_features: int, out_features: int, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.in_features = in_features
        self.out_features = out_features
        self.rmsnorm = SimpleRMSNorm(in_features)

    def forward(self, x: Tensor) -> Tensor:
        # 1. RMSNorm
        x_norm = self.rmsnorm(x)

        # 2. 입력 양자화 (STE)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()

        # 3. 가중치 양자화 (STE)
        # w_q = weight_quant(self.weight) # 함수 사용 Ver

        # 디버깅용, 함수 사용 X Ver
        gamma = self.weight.abs().mean()
        w_q = torch.round(self.weight / (gamma + 1e-5)).clamp_(-1, 1)
        if self.training:
            print(f"[BitLinear] γ: {gamma.item():.4f}")
            print(f"[BitLinear] w_q unique: {w_q.unique(sorted=True)}")  # [-1, 0, 1] 인지 확인


        w_quant = self.weight + (w_q - self.weight).detach()

        # 4. 선형 연산
        return F.linear(x_quant, w_quant, self.bias)