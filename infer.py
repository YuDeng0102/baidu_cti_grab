import os
import sys
import time
import argparse

# --- Competition Triton Cache Setup ---
# Set the triton cache directory to a local folder so that the compiled kernels 
# can be submitted with the code, completely eliminating JIT compilation overhead.
os.environ["TRITON_CACHE_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "triton_cache")
import torch
import contextlib
from pathlib import Path
from tqdm import tqdm

from Utils.profiler import record_scope, export_profiler_csv
from quantization.quantization_utils import apply_quantization_patch
from Utils.data_utils import (
    _detect_has_clk,
    load_sample_files,
    load_logids_from_file,
    CTRUserDataset,
    make_collate_fn,
    move_batch_to_device
)
from models import (
    RepEncoder,
    scaled_dot_product,
    Expert,
    TopKGate,
    SMoE,
    TransformerEncoder,
    CTRModel,
    load_model
)
from metrics import _cal_score, _read_label, _read_predict

# 兼容官方评测要求
CTRTestSeqDataset = CTRUserDataset


def main():
    import io
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint 文件路径，默认使用同目录下的 ckpt.pt')
    parser.add_argument('--profile-batches', type=int, default=0,
                        help='使用 torch.profiler 分析前 N 个 batch；0 表示正常完整推理')
    parser.add_argument('--profile-dir', type=str, default='profiler_traces',
                        help='torch.profiler trace 输出目录，仅在 --profile-batches > 0 时生效')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['fp32', 'bf16', 'fp16'],
                        help='模型推理 dtype，默认 bf16')
    parser.add_argument('--attn-mode', type=str, default='flash_varlen', choices=['sdpa', 'flash_varlen'],
                        help='attention 实现，默认 sdpa；flash_varlen 需要 flash-attn 且 dtype 为 bf16/fp16')
    parser.add_argument('--rep-fuse-mode', type=str, default='torch',
                        choices=['torch', 'cat_free_ref', 'triton_ln_linear'],
                        help='RepEncoder fuse 实现；cat_free_ref 是等价验证用 reference，默认 torch')
    parser.add_argument('--quantized', action='store_true',
                        help='使用量化模型结构')
    args = parser.parse_args()
    import Utils.profiler
    Utils.profiler.PROFILE_SCOPES = args.profile_batches > 0

    cur_path = Path(__file__).parent.absolute()
    ref_dir = cur_path / "dataset"
    history_dir = ref_dir / "history"
    input_file = ref_dir / "test.csv"
    output_file = Path("predict.txt")
    label_file = ref_dir / "label_data.txt"

    # ----- 数据加载，优先从缓存读取 -----
    MAX_SHARD_BYTES = 2 * 1024 * 1024 * 1024  # 2GB per shard
    batches_cache_dir = ref_dir / "cached_batches"

    shard_files = []
    total_batches = 0
    if batches_cache_dir.exists() and any(batches_cache_dir.glob('shard_*.pt')):
        print(f'[INFO] loading cached batch shards from {batches_cache_dir}')
        shard_files = sorted(batches_cache_dir.glob('shard_*.pt'),
                             key=lambda p: int(p.stem.split('_')[1]))
        for sf in shard_files:
            shard_batches = torch.load(sf, weights_only=False)
            total_batches += len(shard_batches)
            print(f'[INFO] loaded metadata for {len(shard_batches)} batches from {sf.name}')
            del shard_batches
        import gc
        gc.collect()
        print(f'[INFO] loaded {total_batches} cached batches metadata total from {len(shard_files)} shards')
    else:
        print("[INFO] start loading data from CSV")
        history_files = (
            sorted(history_dir.glob("*.csv")) if history_dir.exists() else []
        )
        all_files = history_files + [input_file]

        item_dict, user_seq = load_sample_files(sample_files_list=all_files)
        test_pred_logids = load_logids_from_file(input_file)
        print(f"[INFO] Test pred logids count: {len(test_pred_logids)}")

        max_feasign_per_slot = {1: 2}
        test_dataset = CTRUserDataset(
            item_dict,
            user_seq,
            max_feasign_per_slot=max_feasign_per_slot,
            pred_logids=test_pred_logids,
        )
        print(
            f"[INFO] num_users={test_dataset.num_users}, "
            f"total_samples={test_dataset.total_samples}, "
            f"pred_samples={len(test_pred_logids)}, "
            f"max_sign_id={test_dataset.max_sign_id}"
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=50,
            shuffle=False,
            num_workers=0,
            collate_fn=make_collate_fn(test_dataset.max_slot_id),
        )

        # 收集 batches 并按分片缓存
        print('[INFO] collecting batches and saving sharded cache...')
        batches_cache_dir.mkdir(parents=True, exist_ok=True)
        shard_idx = 0
        current_shard = []
        current_size = 0
        for batch in test_loader:
            total_batches += 1
            buf = io.BytesIO()
            torch.save(batch, buf)
            batch_size_bytes = buf.tell()
            if current_shard and current_size + batch_size_bytes > MAX_SHARD_BYTES:
                shard_path = batches_cache_dir / f"shard_{shard_idx:04d}.pt"
                torch.save(current_shard, shard_path)
                print(
                    f"[INFO] saved shard {shard_path.name}: {len(current_shard)} batches, "
                    f"~{current_size / 1024**3:.2f}GB"
                )
                shard_idx += 1
                current_shard = []
                current_size = 0
            current_shard.append(batch)
            current_size += batch_size_bytes
        if current_shard:
            shard_path = batches_cache_dir / f"shard_{shard_idx:04d}.pt"
            torch.save(current_shard, shard_path)
            print(
                f"[INFO] saved shard {shard_path.name}: {len(current_shard)} batches, "
                f"~{current_size / 1024**3:.2f}GB"
            )
            shard_idx += 1
        print(f'[INFO] saved {total_batches} batches to {shard_idx} shards in {batches_cache_dir}')
        shard_files = sorted(batches_cache_dir.glob('shard_*.pt'),
                             key=lambda p: int(p.stem.split('_')[1]))
        del current_shard
        import gc
        gc.collect()

    print("[INFO] data loading done")

    def iter_batches():
        for sf in shard_files:
            shard_batches = torch.load(sf, weights_only=False)
            for batch in shard_batches:
                yield batch
            del shard_batches
            import gc
            gc.collect()

    # ----- 加载模型 -----
    model, dev = load_model(
        ckpt_path=args.ckpt,
        dtype=args.dtype,
        attn_mode=args.attn_mode,
        rep_fuse_mode=args.rep_fuse_mode,
        quantized=args.quantized,
    )

    # ----- 推理 -----
    print("*" * 20 + " start inference " + "*" * 20)
    all_logids = []
    all_probs = []
    time_sum = 0.0

    def infer_one_batch(batch):
        batch = move_batch_to_device(batch, dev)
        pred_mask = batch["pred_mask"].bool()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.time()
        logits, moe_loss = model(batch)
        logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t_start

        masked_logids = batch["logid"][pred_mask].cpu().tolist()
        masked_probs = probs[pred_mask].cpu().tolist()
        return masked_logids, masked_probs, elapsed

    if args.profile_batches > 0:
        from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

        activities = [ProfilerActivity.CPU]
        if dev.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        profile_batches = min(args.profile_batches, total_batches)
        profile_dir = Path(args.profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        if warmup_batches > 0:
            print(f"[INFO] warming up {warmup_batches} batches before profiling")
            with torch.inference_mode():
                for batch in tqdm(all_batches[:warmup_batches], desc="Warmup"):
                    infer_one_batch(batch)
        if profile_batches <= 0:
            print(
                "[WARNING] no batches left to profile after warmup, skipping profiler"
            )
            return None

        profile_slice = all_batches[warmup_batches : warmup_batches + profile_batches]
        print(
            f"[INFO] profiling {profile_batches} batches after {warmup_batches} warmup batches"
        )
        print(f"[INFO] writing profiler traces to {profile_dir.absolute()}")

        profile_batches_list = []
        for batch in iter_batches():
            profile_batches_list.append(batch)
            if len(profile_batches_list) >= profile_batches:
                break

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            on_trace_ready=None,
        ) as prof:
            with torch.inference_mode():
                for batch in tqdm(profile_batches_list, desc="Profile"):
                    with record_scope("infer_one_batch"):
                        masked_logids, masked_probs, elapsed = infer_one_batch(batch)
                    time_sum += elapsed
                    all_logids.extend(masked_logids)
                    all_probs.extend(masked_probs)
                    prof.step()

        sort_by = "cuda_time_total" if dev.type == "cuda" else "cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_by, row_limit=40))
        csv_path = profile_dir / "key_averages.csv"
        export_profiler_csv(prof, csv_path)
        print(f"[INFO] profiled inference time: {round(time_sum, 4)}s")
        print(f"[INFO] profiler key averages written to {csv_path}")
        print(f"[INFO] open with: tensorboard --logdir {profile_dir}")
        return None

    with torch.inference_mode():
        for batch in tqdm(iter_batches(), total=total_batches, desc="Inference"):
            masked_logids, masked_probs, elapsed = infer_one_batch(batch)
            time_sum += elapsed
            all_logids.extend(masked_logids)
            all_probs.extend(masked_probs)

    print(f"[INFO] inference time: {round(time_sum, 4)}s")
    print("*" * 20 + " end inference " + "*" * 20)

    # ----- 按 test.csv 顺序写预测文件 -----
    logid_to_prob = dict(zip(all_logids, all_probs))
    test_logids_in_order = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                test_logids_in_order.append(int(line.split(",")[0]))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for logid in test_logids_in_order:
            f.write(f"{logid_to_prob[logid]}\n")
    print(
        f"[INFO] predictions written to {output_file}, total: {len(test_logids_in_order)}"
    )

    # ----- 打分 -----
    if label_file.exists():
        result = _cal_score(output_file, label_file, default_latency=time_sum)
        print(f"[INFO] AUC:            {result['auc']:.6f}")
        print(f"[INFO] PCOC:           {result['pcoc']:.6f}")
        print(f"[INFO] Latency:        {result['latency']:.4f}s")
        print(f"[INFO] score_latency:  {result['score_latency']:.6f}")
        print(f"[INFO] score_model:    {result['score_model']:.6f}")
        print(f"[INFO] score_all:      {result['score_all']:.6f}")
        return result
    else:
        print(f"[WARNING] label file {label_file} not found, skipping scoring")
        return None


if __name__ == "__main__":
    main()
