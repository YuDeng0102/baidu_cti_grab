import torch
import time

B, S, k, D = 1, 4500, 2, 512
E = 8

x_flat = torch.randn(B * S, D, device='cuda')
route_expert = torch.randint(0, E, (B * S * k,), device='cuda')
route_token = torch.arange(B * S, device='cuda').repeat_interleave(k)

sort_order = torch.argsort(route_expert)
sorted_expert = route_expert[sort_order]
sorted_token = route_token[sort_order]

counts = torch.bincount(sorted_expert, minlength=E)
starts = torch.cumsum(counts, dim=0) - counts
position_in_expert = torch.arange(sorted_expert.numel(), device='cuda') - torch.repeat_interleave(starts, counts)
max_count = int(counts.max().item())

def orig():
    sorted_x = x_flat[sorted_token]
    packed_x = x_flat.new_zeros((E, max_count, D))
    packed_x[sorted_expert, position_in_expert] = sorted_x
    return packed_x

def fused():
    packed_x = x_flat.new_zeros((E, max_count, D))
    packed_x[sorted_expert, position_in_expert] = x_flat[sorted_token]
    return packed_x

# Warmup
orig()
fused()

torch.cuda.synchronize()
t0 = time.time()
for _ in range(100): orig()
torch.cuda.synchronize()
t1 = time.time()
for _ in range(100): fused()
torch.cuda.synchronize()
t2 = time.time()

print("Orig:", (t1 - t0) * 1000)
print("Fused:", (t2 - t1) * 1000)
