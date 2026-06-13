import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_quantization_patch():
    _quant_dir = os.path.dirname(os.path.abspath(__file__))
    if _quant_dir not in sys.path:
        sys.path.append(_quant_dir)
    from w8a8_triton_kernels import w8a8_linear
    from w8a8_preprocess import QuantizedWeight

    class StaticQuantizedLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.use_quant = in_features >= 14336
            if self.use_quant:
                self.register_buffer("qweight_t", torch.zeros((in_features, out_features), dtype=torch.int8))
                self.register_buffer("weight_scale", torch.zeros((out_features,), dtype=torch.float32))
                self.register_buffer("static_act_scale", torch.zeros((1,), dtype=torch.float32))
                if bias:
                    self.register_buffer("bias", torch.zeros((out_features,), dtype=dtype if dtype is not None else torch.float32))
                else:
                    self.register_buffer("bias", None)
            else:
                self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype if dtype is not None else torch.float32))
                if bias:
                    self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype if dtype is not None else torch.float32))
                else:
                    self.register_parameter('bias', None)

        def forward(self, input):
            if self.use_quant:
                # Shape-based Dispatch: if M is small AND it's a smaller layer, fallback to bf16.
                # (For K=14336, benchmark proves W8A8 is ALWAYS faster regardless of M, so we never fallback)
                M = input.numel() // input.shape[-1]
                if M < 3000 and self.in_features < 14336:
                    weight_bf16 = (self.qweight_t.float() * self.weight_scale.unsqueeze(1)).to(input.dtype)
                    bias = self.bias.to(input.dtype) if self.bias is not None else None
                    return F.linear(input, weight_bf16, bias)
                else:
                    qw = QuantizedWeight(qweight_t=self.qweight_t, scale=self.weight_scale, fp_weight=torch.empty(0, device=input.device))
                    return w8a8_linear(input, qw, bias=self.bias, act_mode="static", static_act_scale=self.static_act_scale)
            else:
                return F.linear(input, self.weight, self.bias)

    nn.Linear = StaticQuantizedLinear
    print("[INFO] Applied W8A8 Quantization Patch (nn.Linear -> StaticQuantizedLinear)")

