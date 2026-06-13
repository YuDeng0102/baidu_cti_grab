#!/bin/bash

OUTPUT_ZIP="submission.zip"

echo "[INFO] 正在打包代码到 ${OUTPUT_ZIP} ..."

# 如果已存在同名压缩包，先删除
if [ -f "$OUTPUT_ZIP" ]; then
    rm "$OUTPUT_ZIP"
fi

# 使用 zip 打包当前目录 (.) 
# -r 表示递归
# -x 用于排除特定文件和文件夹
zip -r "$OUTPUT_ZIP" . \
    -x "dataset/*" \
    -x "ckpt.pt" \
    -x "predict.txt" \
    -x "profiler_traces/*" \
    -x "*/__pycache__/*" \
    -x "*.pyc" \
    -x ".git/*" \
    -x ".gemini/*" \
    -x "*.zip" \
    -x "triton_cache/*" \
    -x "scratch"/* \

echo "[INFO] 打包完成！你可以提交 ${OUTPUT_ZIP} 了。"
