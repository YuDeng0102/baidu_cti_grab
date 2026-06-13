import triton
import triton.language as tl
import torch

@triton.jit(do_not_specialize=["M", "a_ptr"])
def kernel(a_ptr, M, BLOCK_M: tl.constexpr):
    pass

print(kernel.do_not_specialize)
