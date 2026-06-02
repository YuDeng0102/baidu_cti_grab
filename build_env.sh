#!/bin/bash
TARGET=flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
cd ..
wget https://ghfast.top/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$TARGET
# pip install $TARGET
/home/aistudio/external-libraries/bin/uv pip install ./$TARGET --target /home/aistudio/libraries

echo "build env succeess"
