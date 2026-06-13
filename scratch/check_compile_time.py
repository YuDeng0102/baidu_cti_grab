import torch
import triton
import sys
import os

_quant_dir = os.path.dirname(os.path.abspath(__file__)) + "/../quantization"
if _quant_dir not in sys.path:
    sys.path.append(_quant_dir)
from w8a8_preprocess import QuantizedWeight
from w8a8_triton_kernels import w8a8_linear

M, K, N = 50, 14336, 512

a = torch.randn((M, K), dtype=torch.bfloat16, device='cuda')
w_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device='cuda').t()
scale_w = torch.randn((N,), dtype=torch.float32, device='cuda')
bias = torch.randn((N,), dtype=torch.bfloat16, device='cuda')
static_act_scale = torch.tensor([1.0], device='cuda')

qw = QuantizedWeight(qweight_t=w_int8, scale=scale_w, fp_weight=torch.empty(0, device='cuda'))

def run_w8a8():
    return w8a8_linear(a, qw, bias=bias, act_mode="static", static_act_scale=static_act_scale)

# Measure first run time (compilation + execution)
torch.cuda.synchronize()
import time
t0 = time.time()
run_w8a8()
torch.cuda.synchronize()
t1 = time.time()

print(f"First run (compilation included): {(t1 - t0) * 1000:.2f} ms")

# Measure execution time
torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for _ in range(100):
    run_w8a8()
end_event.record()
torch.cuda.synchronize()
print(f"Subsequent run (avg): {start_event.elapsed_time(end_event) / 100:.2f} ms")
