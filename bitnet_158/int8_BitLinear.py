import torch
import math
from torch import nn, Tensor
from zeta.nn.modules.simple_rmsnorm import SimpleRMSNorm
import torch.nn.functional as F



def activation_quant(x: Tensor):
    """메모리 효율적인 활성화 양자화"""
    scale = 127.0 / x.detach().abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y + (x - x.detach())  # STE (Straight-Through Estimator)


def weight_quant(w: Tensor):
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign() * scale
    return u


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, debug=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.debug = debug  # 디버깅 모드 활성화 플래그
        
        # 32비트 가중치 (학습용)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # 8비트 가중치 (추론용)
        self.register_buffer('weight_int8', torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 초기 양자화
        self.requantize_weight()
        
        # 디버깅 카운터
        self.forward_count = 0
    
    def requantize_weight(self):
        """가중치 재양자화 메서드"""
        scale = 127.0 / self.weight.data.abs().max().clamp(min=1e-5)
        self.weight_int8.data = (self.weight.data * scale).round().clamp(-128, 127).to(torch.int8)
        self.weight_scale.data = scale
        
        # 디버깅 출력
        if self.debug:
            print("\n[BitLinear Weight Quantization Debug]")
            print(f"Scale: {scale.item():.6f}")
            
            # 원본 가중치 통계
            w_flat = self.weight.data.flatten()
            print(f"Original weight: min={w_flat.min().item():.4f}, max={w_flat.max().item():.4f}, "
                  f"mean={w_flat.mean().item():.4f}, std={w_flat.std().item():.4f}")
            
            # 양자화된 가중치 통계
            wq_flat = self.weight_int8.float().flatten()
            print(f"Quantized weight: min={wq_flat.min().item():.4f}, max={wq_flat.max().item():.4f}, "
                  f"mean={wq_flat.mean().item():.4f}, std={wq_flat.std().item():.4f}")
            
            # -1, 0, 1 값 비율 계산
            total = wq_flat.numel()
            neg_count = (wq_flat < -0.5).sum().item()
            zero_count = (wq_flat.abs() <= 0.5).sum().item()
            pos_count = (wq_flat > 0.5).sum().item()
            
            print(f"Ternary distribution: -1={neg_count/total*100:.2f}%, "
                  f"0={zero_count/total*100:.2f}%, "
                  f"+1={pos_count/total*100:.2f}%")
    
    def forward(self, x):
        # 8비트 → 32비트 디양자화
        weight_dequant = self.weight_int8.float() * self.weight_scale
        
        # 활성화 양자화 (STE 적용)
        x_norm = SimpleRMSNorm(self.in_features)(x)
        x_quant = activation_quant(x_norm)
        
        # 디버깅 출력 (처음 3번만 출력)
        if self.debug and self.forward_count < 3:
            print("\n[BitLinear Forward Debug]")
            print(f"Forward call #{self.forward_count + 1}")
            
            # 입력 통계
            print(f"Input: shape={x.shape}, min={x.min().item():.4f}, "
                  f"max={x.max().item():.4f}, mean={x.mean().item():.4f}")
            
            # 정규화 후 통계
            print(f"After RMSNorm: min={x_norm.min().item():.4f}, "
                  f"max={x_norm.max().item():.4f}, mean={x_norm.mean().item():.4f}")
            
            # 양자화 후 통계
            xq_flat = x_quant.flatten()
            print(f"After Activation Quant: min={xq_flat.min().item():.4f}, "
                  f"max={xq_flat.max().item():.4f}, mean={xq_flat.mean().item():.4f}")
            
            # -1, 0, 1 값 비율 계산
            total = xq_flat.numel()
            neg_count = (xq_flat < -0.5).sum().item()
            zero_count = (xq_flat.abs() <= 0.5).sum().item()
            pos_count = (xq_flat > 0.5).sum().item()
            
            print(f"Ternary distribution: -1={neg_count/total*100:.2f}%, "
                  f"0={zero_count/total*100:.2f}%, "
                  f"+1={pos_count/total*100:.2f}%")
            
            # 가중치 디양자화 후 통계
            w_deq_flat = weight_dequant.flatten()
            print(f"Dequantized Weight: min={w_deq_flat.min().item():.4f}, "
                  f"max={w_deq_flat.max().item():.4f}, mean={w_deq_flat.mean().item():.4f}")
            
            self.forward_count += 1
        
        return F.linear(x_quant, weight_dequant, self.bias)