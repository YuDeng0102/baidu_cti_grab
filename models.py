import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from Utils.profiler import record_scope
from quantization.quantization_utils import apply_quantization_patch
from kernels.rep_encoder_fusion import cat_free_layernorm_linear, layernorm_linear_triton

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
except ImportError:
    flash_attn_varlen_qkvpacked_func = None

class RepEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, padding_idx=0, slot_num=0, d_model=0, fuse_mode="torch"):
        super().__init__()
        self.emb = nn.EmbeddingBag(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=padding_idx, mode='sum')
        self.emb_dim = emb_dim
        self.slot_num = slot_num
        self.fuse_mode = fuse_mode
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
                    res = self.emb(values, offsets[:-1])
                    pooled_embs.append(res)
            with record_scope("RepEncoder/fuse"):
                if self.fuse_mode == "cat_free_ref":
                    rep_emb = cat_free_layernorm_linear(pooled_embs, self.input_norm, self.linear)
                elif self.fuse_mode == "triton_ln_linear":
                    fused_embs = torch.cat(pooled_embs, dim=1)
                    rep_emb = layernorm_linear_triton(fused_embs, self.input_norm, self.linear)
                else:
                    with record_scope("RepEncoder/fuse/cat"):
                        fused_embs = torch.cat(pooled_embs, dim=1)
                    with record_scope("RepEncoder/fuse/norm"):
                        norm_emb = self.input_norm(fused_embs)
                    with record_scope("RepEncoder/fuse/linear"):
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
        self._stacked_expert_cache = None

    def _get_stacked_expert_params(self, device, dtype):
        cache = self._stacked_expert_cache
        if cache is not None:
            cached_device = cache["w1"].device
            cached_dtype = cache["w1"].dtype
            if cached_device == device and cached_dtype == dtype:
                return cache

        def get_w(layer):
            if hasattr(layer, "use_quant") and layer.use_quant:
                return (layer.qweight_t.to(dtype) * layer.weight_scale).contiguous()
            return layer.weight.transpose(0, 1).contiguous()

        w1 = torch.stack([get_w(expert.fc1) for expert in self.experts], dim=0)
        b1 = torch.stack([expert.fc1.bias for expert in self.experts], dim=0)
        w2 = torch.stack([get_w(expert.fc2) for expert in self.experts], dim=0)
        b2 = torch.stack([expert.fc2.bias for expert in self.experts], dim=0)

        cache = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
        self._stacked_expert_cache = cache
        return cache

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
            route_expert = idx_flat.reshape(-1)
            route_weight = score_flat.reshape(-1)
            route_token = torch.arange(B * S, device=x.device, dtype=torch.long).repeat_interleave(self.k)

            sort_order = torch.argsort(route_expert, stable=True)
            sorted_expert = route_expert[sort_order]
            sorted_weight = route_weight[sort_order]
            sorted_token = route_token[sort_order]
            sorted_x = x_flat[sorted_token]

            counts = torch.bincount(sorted_expert, minlength=self.num_experts)
            if sorted_expert.numel() > 0:
                starts = torch.cumsum(counts, dim=0) - counts
                position_in_expert = torch.arange(sorted_expert.numel(), device=x.device, dtype=torch.long)
                position_in_expert = position_in_expert - torch.repeat_interleave(starts, counts)
                max_count = int(counts.max().item())

                packed_x = x.new_zeros((self.num_experts, max_count, D))
                packed_x[sorted_expert, position_in_expert] = sorted_x

                params = self._get_stacked_expert_params(device=x.device, dtype=x.dtype)
                with record_scope("SMoE/grouped_experts"):
                    hidden = torch.bmm(packed_x, params["w1"]) + params["b1"].unsqueeze(1)
                    hidden = F.relu(hidden)
                    packed_out = torch.bmm(hidden, params["w2"]) + params["b2"].unsqueeze(1)

                routed_out = packed_out[sorted_expert, position_in_expert]
                weighted_out = routed_out * sorted_weight.unsqueeze(-1)
                out.reshape(-1, D).index_add_(0, sorted_token, weighted_out)

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

def load_model(device='cuda:0', ckpt_path=None, dtype='bf16', attn_mode='flash_varlen', rep_fuse_mode='torch', quantized=False):
    """加载模型并返回，供 evaluation.py 调用。

    Args:
        device: 推理设备（默认 'cuda:0'）
        ckpt_path: checkpoint 文件路径，默认使用 infer.py 同目录下的 ckpt.pt
        dtype: 模型推理 dtype，可选 fp32、bf16、fp16
        attn_mode: attention 实现，可选 sdpa、flash_varlen
        rep_fuse_mode: RepEncoder fuse 实现，可选 torch、cat_free_ref、triton_ln_linear
        quantized: 是否应用 W8A8 量化网络结构，默认 False

    Returns:
        (model, device) 元组
    """
    if quantized:
        apply_quantization_patch()
    emb_dim = 512
    slot_num = 28
    vocab_size = 5000000
    d_model = 512
    n_heads = 8
    num_layers = 8
    dim_ff = 1024

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    with torch.device(dev):
        rep_encoder = RepEncoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            padding_idx=0,
            slot_num=slot_num,
            d_model=d_model,
            fuse_mode=rep_fuse_mode,
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
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[INFO] Loaded checkpoint from {ckpt_path} (epoch={ckpt.get('epoch', '?')})")
        del ckpt
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
