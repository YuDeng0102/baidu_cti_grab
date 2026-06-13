import torch
import triton
import sys
import os

_quant_dir = os.path.dirname(os.path.abspath(__file__)) + "/../quantization"
if _quant_dir not in sys.path:
    sys.path.append(_quant_dir)
from w8a8_preprocess import QuantizedWeight
from w8a8_triton_kernels import w8a8_linear

M, K, N = 4500, 14336, 512

a = torch.randn((M, K), dtype=torch.bfloat16, device='cuda')
w = torch.randn((N, K), dtype=torch.bfloat16, device='cuda').t()

# bf16
def run_bf16():
    return torch.matmul(a, w)

# w8a8
a_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='cuda')
w_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device='cuda').t()
scale_a = torch.randn((M, 1), dtype=torch.float32, device='cuda')
scale_w = torch.randn((N,), dtype=torch.float32, device='cuda')
bias = torch.randn((N,), dtype=torch.bfloat16, device='cuda')

qw = QuantizedWeight(qweight_t=w_int8, scale=scale_w, fp_weight=torch.empty(0, device='cuda'))
def run_w8a8():
    return w8a8_linear(a, qw, bias=bias, act_mode="static", static_act_scale=torch.tensor([1.0], device='cuda'))

# warmup
run_bf16()
run_w8a8()

torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for _ in range(100): run_bf16()
end_event.record()
torch.cuda.synchronize()
bf16_time = start_event.elapsed_time(end_event) / 100

start_event.record()
for _ in range(100): run_w8a8()
end_event.record()
torch.cuda.synchronize()
w8a8_time = start_event.elapsed_time(end_event) / 100

print(f"M={M} K={K} N={N}")
print(f"BF16 Time: {bf16_time:.4f} ms")
print(f"W8A8 Time: {w8a8_time:.4f} ms")
print(f"Speedup: {bf16_time / w8a8_time:.2f}x")
