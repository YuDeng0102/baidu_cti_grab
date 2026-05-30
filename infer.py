import sys
import os
import csv

# 获取当前环境脚本所在目录或指定绝对路径
if os.path.exists("../libraries"):
    lib_path = os.path.abspath("../libraries")
    sys.path.append(lib_path)

import math
import argparse
from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
except Exception:
    flash_attn_varlen_qkvpacked_func = None


PROFILE_SCOPES = False


def record_scope(name):
    if PROFILE_SCOPES:
        return torch.profiler.record_function(name)
    return nullcontext()


def export_profiler_csv(prof, output_path):
    rows = prof.key_averages()
    fieldnames = [
        "name",
        "calls",
        "self_cpu_time_total_us",
        "cpu_time_total_us",
        "cpu_time_avg_us",
        "self_cuda_time_total_us",
        "cuda_time_total_us",
        "cuda_time_avg_us",
        "cpu_memory_usage_bytes",
        "self_cpu_memory_usage_bytes",
        "cuda_memory_usage_bytes",
        "self_cuda_memory_usage_bytes",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            calls = row.count
            self_cuda_time_total = getattr(row, "self_cuda_time_total",
                                           getattr(row, "self_device_time_total", 0))
            cuda_time_total = getattr(row, "cuda_time_total",
                                      getattr(row, "device_time_total", 0))
            cuda_memory_usage = getattr(row, "cuda_memory_usage",
                                        getattr(row, "device_memory_usage", 0))
            self_cuda_memory_usage = getattr(row, "self_cuda_memory_usage",
                                             getattr(row, "self_device_memory_usage", 0))
            writer.writerow({
                "name": row.key,
                "calls": calls,
                "self_cpu_time_total_us": row.self_cpu_time_total,
                "cpu_time_total_us": row.cpu_time_total,
                "cpu_time_avg_us": row.cpu_time_total / calls if calls else 0,
                "self_cuda_time_total_us": self_cuda_time_total,
                "cuda_time_total_us": cuda_time_total,
                "cuda_time_avg_us": cuda_time_total / calls if calls else 0,
                "cpu_memory_usage_bytes": row.cpu_memory_usage,
                "self_cpu_memory_usage_bytes": row.self_cpu_memory_usage,
                "cuda_memory_usage_bytes": cuda_memory_usage,
                "self_cuda_memory_usage_bytes": self_cuda_memory_usage,
            })


# ============================================================
# 数据加载（来自 train/dataset.py）
# ============================================================

def _detect_has_clk(file_path):
    """检测 CSV 文件是否包含 clk 列（5列 vs 4列格式）。
    5列格式: logid,userid,adid,clk,timestamp,sign:slot...
    4列格式: logid,userid,adid,timestamp,sign:slot...
    通过第5个字段是否包含 ':' 来判断：有 ':' 说明已经是 sign:slot，即无 clk 列。
    """
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 5:
                return ':' not in parts[4]
            return False
    return False


def load_sample_files(sample_files_list):
    """加载 CSV sample 文件，返回 item_dict 和 user_seq。
    自动检测每个文件是 5列（含clk）还是 4列（无clk）格式。
    """
    sample_files = sorted([Path(f) for f in sample_files_list])
    print(f'[INFO] loading {len(sample_files)} files: {[str(f) for f in sample_files]}')

    item_dict = {}
    user_logs = defaultdict(list)

    for sample_file in tqdm(sample_files, desc='Loading sample files'):
        has_clk = _detect_has_clk(sample_file)
        min_parts = 5 if has_clk else 4
        print(f'  {sample_file.name}: has_clk={has_clk}')

        with open(sample_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < min_parts:
                    continue

                logid = int(parts[0])
                userid = int(parts[1])
                adid = int(parts[2])

                if has_clk:
                    clk = int(parts[3])
                    timestamp = int(parts[4])
                    feat_start = 5
                else:
                    clk = 0
                    timestamp = int(parts[3])
                    feat_start = 4

                signs = []
                slots = []
                for pair in parts[feat_start:]:
                    if ':' in pair:
                        s, sl = pair.split(':', 1)
                        signs.append(int(s))
                        slots.append(int(sl))

                item_dict[logid] = {
                    'logid': logid,
                    'userid': userid,
                    'adid': adid,
                    'clk': clk,
                    'timestamp': timestamp,
                    'signs': np.array(signs, dtype=np.int64),
                    'slots': np.array(slots, dtype=np.int64),
                }
                user_logs[userid].append((timestamp, logid))

    user_seq = {}
    for userid, logs in user_logs.items():
        logs.sort(key=lambda x: x[0])
        user_seq[userid] = [logid for _, logid in logs]

    print(f'[INFO] loaded {len(item_dict)} records, {len(user_seq)} users')
    return item_dict, user_seq


def load_logids_from_file(file_path):
    """快速读取一个 sample 文件中的所有 logid"""
    logids = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            comma = line.index(',')
            logids.add(int(line[:comma]))
    return logids


class CTRUserDataset(Dataset):
    """按用户组织的 CTR 数据集"""

    def __init__(self, item_dict, user_seq=None, max_feasign_per_slot=None, pred_logids=None):
        super().__init__()
        self.item_dict = item_dict
        self.user_seq = user_seq if user_seq else {}
        self.max_feasign_per_slot = max_feasign_per_slot
        self.pred_logids = pred_logids if pred_logids is not None else set()

        self.user_items = defaultdict(list)
        for logid, rec in item_dict.items():
            userid = rec['userid']
            feasign = defaultdict(list)
            for slot, sign in zip(rec['slots'].tolist(), rec['signs'].tolist()):
                feasign[slot].append(sign)
            if max_feasign_per_slot is not None:
                feasign = {slot: signs[:max_feasign_per_slot[slot]]
                           if max_feasign_per_slot.get(slot, -1) != -1 else signs
                           for slot, signs in feasign.items()}
            feasign = dict(feasign)
            label = rec['clk']
            self.user_items[userid].append((logid, feasign, label))

        self.user_ids = sorted(self.user_items.keys())
        self.num_users = len(self.user_ids)
        self.total_samples = len(item_dict)

        all_signs = set()
        for rec in item_dict.values():
            all_signs.update(rec['signs'].tolist())
        self.max_slot_id = 28
        self.max_sign_id = max(all_signs) if all_signs else 0

    def __len__(self):
        return self.num_users

    def __getitem__(self, index):
        userid = self.user_ids[index]
        items = self.user_items[userid]

        if self.user_seq and userid in self.user_seq:
            seq_order = {logid: i for i, logid in enumerate(self.user_seq[userid])}
            items.sort(key=lambda x: seq_order.get(x[0], x[0]))
        else:
            items.sort(key=lambda x: x[0])

        feasigns = []
        labels = []
        logids = []
        for logid, feasign, label in items:
            logids.append(logid)
            feasigns.append(feasign)
            labels.append(label)

        return {
            'userid': userid,
            'logids': logids,
            'feasigns': feasigns,
            'labels': labels,
            'pred_mask': [1 if logid in self.pred_logids else 0 for logid in logids],
        }


def make_collate_fn(max_slot_id):
    def collate_user_batch(batch):
        all_userids = []
        all_logids = []
        all_labels = []
        all_pred_masks = []
        all_feasigns = []
        user_offsets = [0]

        for item in batch:
            for i, logid in enumerate(item['logids']):
                all_userids.append(item['userid'])
                all_logids.append(logid)
                all_labels.append(item['labels'][i])
                all_pred_masks.append(item['pred_mask'][i])
                all_feasigns.append(item['feasigns'][i])
            user_offsets.append(len(all_labels))

        slot_data = {}
        for slot in range(1, max_slot_id + 1):
            values = []
            offsets = [0]
            for feasign in all_feasigns:
                if slot in feasign:
                    values.extend(feasign[slot])
                offsets.append(len(values))
            slot_data[slot] = (
                torch.tensor(values, dtype=torch.long),
                torch.tensor(offsets, dtype=torch.long),
            )

        result = {
            'userid': torch.tensor(all_userids, dtype=torch.long),
            'logid': torch.tensor(all_logids, dtype=torch.long),
            'label': torch.tensor(all_labels, dtype=torch.float32),
            'pred_mask': torch.tensor(all_pred_masks, dtype=torch.bool),
            'user_offsets': torch.tensor(user_offsets, dtype=torch.long),
        }
        result.update(slot_data)
        return result

    return collate_user_batch


# ============================================================
# 模型定义（来自 main.py）
# ============================================================

def move_batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [move_batch_to_device(x, device) for x in batch]
    elif torch.is_tensor(batch):
        return batch.to(device)
    else:
        return batch


class RepEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, padding_idx=0, slot_num=0, d_model=0):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=padding_idx)
        self.emb_dim = emb_dim
        self.slot_num = slot_num
        self.input_norm = nn.LayerNorm(slot_num * emb_dim)
        self.linear = nn.Linear(in_features=slot_num * emb_dim, out_features=d_model)

    def forward(self, batch):
        with record_scope("RepEncoder"):
            pooled_embs = []
            max_idx = self.emb.num_embeddings - 1
            for i in range(self.slot_num):
                with record_scope(f"RepEncoder/slot_{i + 1:02d}"):
                    values, offsets = batch[i + 1]
                    offsets = offsets.to(values.device)
                    values = values.clamp(0, max_idx)  # 超出 vocab_size 的 sign id 截断，避免越界
                    sign_emb = self.emb(values)
                    res = torch.segment_reduce(sign_emb, reduce='sum', offsets=offsets, initial=0)
                    pooled_embs.append(res)
            with record_scope("RepEncoder/fuse"):
                fused_embs = torch.cat(pooled_embs, dim=1)
                norm_emb = self.input_norm(fused_embs)
                rep_emb = self.linear(norm_emb)
            return rep_emb


def scaled_dot_product(q, k, v, extension):
    mask = None
    if extension is not None and "mask" in extension:
        mask = extension["mask"]
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )


class Expert(nn.Module):
    def __init__(self, d_model, dim_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TopKGate(nn.Module):
    def __init__(self, d_model, num_experts, k=2, noisy_gating=True):
        super().__init__()
        self.w_g = nn.Linear(d_model, num_experts)
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating

    def forward(self, x):
        # x: [B,S,D]
        logits = self.w_g(x)  # [B,S,E]

        if self.noisy_gating and self.training:
            logits = logits + torch.randn_like(logits) * 0.1

        probs = torch.softmax(logits, dim=-1)  # [B,S,E]

        topk_score, topk_idx = torch.topk(probs, self.k, dim=-1)  # [B,S,k]

        return topk_idx, topk_score, probs

class SMoE(nn.Module):
    def __init__(self, d_model, dim_ff, num_experts, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList([
            Expert(d_model, dim_ff) for _ in range(num_experts)
        ])

        self.gate = TopKGate(d_model, num_experts, k=k)

    def forward(self, x):
        with record_scope("SMoE"):
            # x: [B,S,D]
            B, S, D = x.shape

            with record_scope("SMoE/gate"):
                topk_idx, topk_score, probs = self.gate(x)

            out = torch.zeros_like(x)

            # flatten
            x_flat = x.reshape(-1, D)                # [B*S, D]
            idx_flat = topk_idx.reshape(-1, self.k)  # [B*S, k]
            score_flat = topk_score.reshape(-1, self.k)

            for i in range(self.num_experts):
                with record_scope(f"SMoE/expert_{i}"):
                    # 找到被路由到 expert i 的 token
                    mask = (idx_flat == i)  # [B*S, k]

                    if not mask.any():
                        continue

                    # 哪些 token 命中了 expert i
                    token_idx, k_idx = mask.nonzero(as_tuple=True)

                    selected_x = x_flat[token_idx]  # [N, D]

                    expert_out = self.experts[i](selected_x)  # [N, D]

                    weight = score_flat[token_idx, k_idx].unsqueeze(-1)

                    out_flat = out.reshape(-1, D)
                    out_flat[token_idx] += expert_out * weight

            importance = probs.sum(dim=(0,1))  # [E]
            moe_loss = (importance.std() / (importance.mean() + 1e-6))

            return out, moe_loss


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dim_ff, act="relu",
                 attention_fn=scaled_dot_product, attn_mode="sdpa"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_layers = num_layers
        self.attn_mode = attn_mode
        assert d_model % n_heads == 0

        self.qkv_proj = nn.ModuleList([nn.Linear(d_model, 3 * d_model) for _ in range(num_layers)])
        self.out_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.ffn1 = nn.ModuleList([nn.Linear(d_model, dim_ff) for _ in range(num_layers)])
        self.ffn2 = nn.ModuleList([nn.Linear(dim_ff, d_model) for _ in range(num_layers)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.act = getattr(F, act)
        self.attention_fn = attention_fn
        self.moe = nn.ModuleList([
            SMoE(d_model, dim_ff, num_experts=8, k=2)
            for _ in range(num_layers)
        ])
        if self.attn_mode == "flash_varlen" and flash_attn_varlen_qkvpacked_func is None:
            print("[WARNING] flash-attn not available, falling back to sdpa attention")
            self.attn_mode = "sdpa"

    def forward(self, x, extension):
        with record_scope("TransformerEncoder"):
            x = x.unsqueeze(0)
            B, S, D = x.shape

            moe_loss_total = 0.0
            for i in range(self.num_layers):
                with record_scope(f"Transformer/layer_{i}"):
                    residual = x
                    x = self.norm1[i](x)
                    with record_scope(f"Transformer/layer_{i}/attention"):
                        qkv = self.qkv_proj[i](x)
                        if self.attn_mode == "flash_varlen" and qkv.dtype in (torch.float16, torch.bfloat16):
                            qkv = qkv.view(B, S, self.n_heads, 3, self.head_dim)
                            qkv_packed = qkv.squeeze(0).permute(0, 2, 1, 3).contiguous()
                            attn_out = flash_attn_varlen_qkvpacked_func(
                                qkv_packed,
                                extension["cu_seqlens"],
                                extension["max_seqlen"],
                                dropout_p=0.0,
                                causal=True,
                            )
                            attn_out = attn_out.reshape(B, S, D)
                        else:
                            qkv = qkv.view(B, S, self.n_heads, 3 * self.head_dim)
                            qkv = qkv.permute(0, 2, 1, 3)
                            q, k, v = torch.split(qkv, self.head_dim, dim=-1)
                            attn_out = self.attention_fn(q, k, v, extension)
                            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, D)
                    x = residual + self.out_proj[i](attn_out)
                    residual = x
                    x = self.norm2[i](x)

                    with record_scope(f"Transformer/layer_{i}/moe"):
                        moe_out, moe_loss = self.moe[i](x)

                    x = residual + moe_out

                    moe_loss_total = moe_loss_total + moe_loss

            return x, moe_loss_total


class CTRModel(nn.Module):
    def __init__(self, rep_encoder, seq_encoder, d_model):
        super().__init__()
        self.rep_encoder = rep_encoder
        self.seq_encoder = seq_encoder
        self.d_model = d_model
        self.linear = nn.Linear(d_model, 1)

    def get_sequence_causal_mask(self, seq_info):
        lengths = seq_info[1:] - seq_info[:-1]
        lengths = lengths.view(-1)
        indices = torch.cumsum(torch.ones_like(lengths), dim=0) - 1
        result = torch.repeat_interleave(indices, lengths)
        a = result.view(1, -1) - result.view(-1, 1)
        out_mask = torch.tril((a == 0).to(torch.int32)).bool()
        return out_mask

    def forward(self, batch):
        with record_scope("CTRModel"):
            seq_input = self.rep_encoder(batch)
            if getattr(self.seq_encoder, "attn_mode", "sdpa") == "flash_varlen":
                with record_scope("CTRModel/build_flash_varlen_info"):
                    user_offsets = batch["user_offsets"]
                    lengths = user_offsets[1:] - user_offsets[:-1]
                    max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
                    extension = {
                        "cu_seqlens": user_offsets.to(dtype=torch.int32),
                        "max_seqlen": max_seqlen,
                    }
            else:
                with record_scope("CTRModel/build_mask"):
                    seq_mask = self.get_sequence_causal_mask(batch["user_offsets"])
                    extension = {"mask": seq_mask.unsqueeze(0).unsqueeze(0)}
            encoder_output, moe_loss = self.seq_encoder(
                x=seq_input,
                extension=extension,
            )
            with record_scope("CTRModel/pred_head"):
                encoder_output_dim = encoder_output.shape[-1]
                encoder_output = encoder_output.reshape(1, -1, encoder_output_dim).squeeze(0)
                pred = self.linear(encoder_output)
                pred_logits = torch.clamp(pred, min=-15.0, max=15.0)
            return pred_logits, moe_loss


# ============================================================
# 模型加载入口
# ============================================================

def load_model(device='cuda:0', ckpt_path=None, dtype='bf16', attn_mode='flash_varlen'):
    """加载模型并返回，供 evaluation.py 调用。

    Args:
        device: 推理设备（默认 'cuda:0'）
        ckpt_path: checkpoint 文件路径，默认使用 infer.py 同目录下的 ckpt.pt
        dtype: 模型推理 dtype，可选 fp32、bf16、fp16
        attn_mode: attention 实现，可选 sdpa、flash_varlen

    Returns:
        (model, device) 元组
    """
    emb_dim = 512
    slot_num = 28
    vocab_size = 5000000
    d_model = 512
    n_heads = 8
    num_layers = 8
    dim_ff = 1024

    rep_encoder = RepEncoder(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        padding_idx=0,
        slot_num=slot_num,
        d_model=d_model,
    )
    seq_encoder = TransformerEncoder(
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_ff=dim_ff,
        act="relu",
        attn_mode=attn_mode,
    )
    model = CTRModel(rep_encoder, seq_encoder, d_model=d_model)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if seq_encoder.attn_mode == "flash_varlen":
        if dev.type != "cuda":
            print("[WARNING] flash_varlen requested on CPU, falling back to sdpa attention")
            seq_encoder.attn_mode = "sdpa"
        elif dtype == "fp32":
            print("[WARNING] flash_varlen requires bf16/fp16, falling back to sdpa attention for fp32")
            seq_encoder.attn_mode = "sdpa"

    # 加载 checkpoint
    # 若需要加载自定义修改的权重，请修改 479-488行逻辑，强制使用你文件夹中的权重
    # 测评系统默认使用原始官方权重
    if ckpt_path is None:
        ckpt_path = Path(__file__).parent / 'ckpt.pt'
    else:
        ckpt_path = Path(ckpt_path)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[INFO] Loaded checkpoint from {ckpt_path} (epoch={ckpt.get('epoch', '?')})")
    else:
        print(f"[WARNING] Checkpoint {ckpt_path} not found, using random weights")

    if dtype == 'bf16':
        if dev.type != 'cuda':
            print("[WARNING] bf16 requested on CPU, falling back to fp32")
            model.to(dev)
        else:
            model.to(dev, dtype=torch.bfloat16)
    elif dtype == 'fp16':
        if dev.type != 'cuda':
            print("[WARNING] fp16 requested on CPU, falling back to fp32")
            model.to(dev)
        else:
            model.to(dev, dtype=torch.float16)
    else:
        model.to(dev)
    model.eval()
    print(f"[INFO] Model ready. Device: {dev}, dtype: {dtype}, attn_mode: {seq_encoder.attn_mode}")

    return model, dev


# ============================================================
# 打分工具（与 evaluation.py 保持一致）
# ============================================================

def _read_predict(file_path):
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(float(line))
    import numpy as np
    return np.array(predictions)


def _read_label(file_path):
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 4:
                    labels.append(float(parts[3]))
                else:
                    labels.append(float(line))
    import numpy as np
    return np.array(labels)


def _cal_score(predict_file, label_file, default_latency=0.0):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    predictions = _read_predict(predict_file)
    labels = _read_label(label_file)

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print('[WARNING] only one class present in labels, AUC is not defined, returning 0.5')
        auc = 0.5
    else:
        auc = roc_auc_score(labels, predictions)

    mean_pred = np.mean(predictions)
    mean_label = np.mean(labels)
    if mean_label == 0:
        pcoc = 1.0 if mean_pred == 0 else float('inf')
    else:
        pcoc = float(mean_pred / mean_label)

    latency = default_latency
    base_latency = 300
    score_latency = max(0.0, (base_latency - latency) / base_latency) if latency < base_latency else 0.0

    if pcoc < 0.85 or pcoc > 1.15:
        score_model = 0.0
    else:
        score_model = ((auc - 0.65) * 1000 + (0.15 - abs(pcoc - 1)) / 0.15 * 10) / 360

    score_all = score_latency * 70 + score_model * 30

    return {
        'auc': auc,
        'pcoc': pcoc,
        'latency': latency,
        'score_latency': score_latency,
        'score_model': score_model,
        'score_all': score_all,
    }


# ============================================================
# main：直接运行 infer.py 进行测试
# ============================================================

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
    args = parser.parse_args()
    global PROFILE_SCOPES
    PROFILE_SCOPES = args.profile_batches > 0

    cur_path = Path(__file__).parent.absolute()
    ref_dir = cur_path / 'dataset'
    history_dir = ref_dir / 'history'
    input_file = ref_dir / 'test.csv'
    output_file = Path('predict.txt')
    label_file = ref_dir / 'label_data.txt'

    # ----- 数据加载，优先从缓存读取 -----
    MAX_SHARD_BYTES = 2 * 1024 * 1024 * 1024  # 2GB per shard
    batches_cache_dir = ref_dir / 'cached_batches'

    if batches_cache_dir.exists() and any(batches_cache_dir.glob('shard_*.pt')):
        print(f'[INFO] loading cached batch shards from {batches_cache_dir}')
        all_batches = []
        shard_files = sorted(batches_cache_dir.glob('shard_*.pt'),
                             key=lambda p: int(p.stem.split('_')[1]))
        for sf in shard_files:
            shard_batches = torch.load(sf, weights_only=False)
            all_batches.extend(shard_batches)
            print(f'[INFO] loaded {len(shard_batches)} batches from {sf.name}')
        print(f'[INFO] loaded {len(all_batches)} cached batches total from {len(shard_files)} shards')
    else:
        print('[INFO] start loading data from CSV')
        history_files = sorted(history_dir.glob('*.csv')) if history_dir.exists() else []
        all_files = history_files + [input_file]

        item_dict, user_seq = load_sample_files(sample_files_list=all_files)
        test_pred_logids = load_logids_from_file(input_file)
        print(f'[INFO] Test pred logids count: {len(test_pred_logids)}')

        max_feasign_per_slot = {1: 2}
        test_dataset = CTRUserDataset(
            item_dict, user_seq,
            max_feasign_per_slot=max_feasign_per_slot,
            pred_logids=test_pred_logids,
        )
        print(f'[INFO] num_users={test_dataset.num_users}, '
              f'total_samples={test_dataset.total_samples}, '
              f'pred_samples={len(test_pred_logids)}, '
              f'max_sign_id={test_dataset.max_sign_id}')

        test_loader = DataLoader(
            test_dataset,
            batch_size=50,
            shuffle=False,
            num_workers=0,
            collate_fn=make_collate_fn(test_dataset.max_slot_id),
        )

        # 收集 batches 并按分片缓存
        print('[INFO] collecting batches and saving sharded cache...')
        all_batches = [batch for batch in test_loader]

        batches_cache_dir.mkdir(parents=True, exist_ok=True)
        shard_idx = 0
        current_shard = []
        current_size = 0
        for batch in all_batches:
            buf = io.BytesIO()
            torch.save(batch, buf)
            batch_size_bytes = buf.tell()
            if current_shard and current_size + batch_size_bytes > MAX_SHARD_BYTES:
                shard_path = batches_cache_dir / f'shard_{shard_idx:04d}.pt'
                torch.save(current_shard, shard_path)
                print(f'[INFO] saved shard {shard_path.name}: {len(current_shard)} batches, '
                      f'~{current_size / 1024**3:.2f}GB')
                shard_idx += 1
                current_shard = []
                current_size = 0
            current_shard.append(batch)
            current_size += batch_size_bytes
        if current_shard:
            shard_path = batches_cache_dir / f'shard_{shard_idx:04d}.pt'
            torch.save(current_shard, shard_path)
            print(f'[INFO] saved shard {shard_path.name}: {len(current_shard)} batches, '
                  f'~{current_size / 1024**3:.2f}GB')
            shard_idx += 1
        print(f'[INFO] saved {len(all_batches)} batches to {shard_idx} shards in {batches_cache_dir}')

    print('[INFO] data loading done')

    # ----- 加载模型 -----
    model, dev = load_model(ckpt_path=args.ckpt, dtype=args.dtype, attn_mode=args.attn_mode)

    # ----- 推理 -----
    print('*' * 20 + ' start inference ' + '*' * 20)
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
        profile_batches = min(args.profile_batches, len(all_batches))
        profile_dir = Path(args.profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        print(f'[INFO] profiling first {profile_batches} batches')
        print(f'[INFO] writing profiler traces to {profile_dir.absolute()}')

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            on_trace_ready=tensorboard_trace_handler(str(profile_dir)),
        ) as prof:
            with torch.inference_mode():
                for batch in tqdm(all_batches[:profile_batches], desc="Profile"):
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
        print(f'[INFO] profiled inference time: {round(time_sum, 4)}s')
        print(f'[INFO] profiler key averages written to {csv_path}')
        print(f'[INFO] open with: tensorboard --logdir {profile_dir}')
        return None

    with torch.inference_mode():
        for batch in tqdm(all_batches, desc="Inference"):
            masked_logids, masked_probs, elapsed = infer_one_batch(batch)
            time_sum += elapsed
            all_logids.extend(masked_logids)
            all_probs.extend(masked_probs)

    print(f'[INFO] inference time: {round(time_sum, 4)}s')
    print('*' * 20 + ' end inference ' + '*' * 20)

    # ----- 按 test.csv 顺序写预测文件 -----
    logid_to_prob = dict(zip(all_logids, all_probs))
    test_logids_in_order = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                test_logids_in_order.append(int(line.split(',')[0]))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for logid in test_logids_in_order:
            f.write(f"{logid_to_prob[logid]}\n")
    print(f'[INFO] predictions written to {output_file}, total: {len(test_logids_in_order)}')

    # ----- 打分 -----
    if label_file.exists():
        result = _cal_score(output_file, label_file, default_latency=time_sum)
        print(f'[INFO] AUC:            {result["auc"]:.6f}')
        print(f'[INFO] PCOC:           {result["pcoc"]:.6f}')
        print(f'[INFO] Latency:        {result["latency"]:.4f}s')
        print(f'[INFO] score_latency:  {result["score_latency"]:.6f}')
        print(f'[INFO] score_model:    {result["score_model"]:.6f}')
        print(f'[INFO] score_all:      {result["score_all"]:.6f}')
        return result
    else:
        print(f'[WARNING] label file {label_file} not found, skipping scoring')
        return None


if __name__ == '__main__':
    main()
