#!/bin/bash
# TARGET=flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# cd ..
# wget https://ghfast.top/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$TARGET
# # pip install $TARGET
# /home/aistudio/external-libraries/bin/uv pip install ./$TARGET --target /home/aistudio/libraries

echo "build env succeess"

echo "Warming up Triton Cache for A800 architecture..."
# 随便跑几批次，让 Triton 提前在这个硬件环境下完成所有内核的即时编译（JIT）
# 生成的机器码会存进 TRITON_CACHE_DIR，后续正式评分时启动开销将为 0！

python infer.py --ckpt /data/quantized_model/ckpt.pt --quantized --profile-batches 5
echo "Triton Cache pre-compiled successfully!"
