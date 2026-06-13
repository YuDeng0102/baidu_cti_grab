import os
from pathlib import Path
import sys

os.environ["QUANT_MODE"] = "preprocess"
import torch
import torch.nn as nn
from tqdm import tqdm

import argparse

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_repo_root)
import infer
from quantization.w8a8_preprocess import quantize_weight_per_channel

def get_calib_loader(dataset_dir: str):
    dataset_dir = Path(dataset_dir)
    all_files = sorted(list(dataset_dir.glob("*.csv")))
    calib_files = all_files[:5]
    print(f"Using {len(calib_files)} files for calibration.")
    item_dict, user_seq = infer.load_sample_files(calib_files)
    dataset = infer.CTRUserDataset(
        item_dict, user_seq,
        max_feasign_per_slot={1: 2},
        pred_logids=set(item_dict.keys()),
    )
    loader = infer.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=infer.make_collate_fn(dataset.max_slot_id))
    return loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="/root/rivermind-data/cti_model/ckpt.pt")
    parser.add_argument("--out_dir", type=str, default="/root/rivermind-data/quantized_cti_model")
    parser.add_argument("--dataset_dir", type=str, default=os.path.join(_repo_root, "dataset"))
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device
    ckpt_path = args.ckpt_path
    print(f"Loading model from {ckpt_path}...")
    model, _ = infer.load_model(device=device, ckpt_path=ckpt_path, attn_mode="sdpa")
    model.eval()

    target_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.in_features >= 14336:
                target_modules[name] = module

    print(f"Found {len(target_modules)} linear layers to quantize.")

    act_scales = {}
    hooks = []
    
    def get_hook(name):
        def hook(module, input, output):
            x = input[0].detach().float()
            current_max = x.abs().amax().item()
            if name not in act_scales:
                act_scales[name] = current_max
            else:
                act_scales[name] = max(act_scales[name], current_max)
        return hook

    for name, module in target_modules.items():
        hooks.append(module.register_forward_hook(get_hook(name)))

    print("Running calibration batches...")
    loader = get_calib_loader(args.dataset_dir)
    num_batches = 10
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            batch = infer.move_batch_to_device(batch, device)
            model(batch)

    for h in hooks:
        h.remove()

    print("Quantizing weights and building new state dict...")
    state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k.endswith('.weight'):
            mod_name = k[:-7]
            if mod_name in target_modules:
                print(f"Quantizing {mod_name}...")
                qweight = quantize_weight_per_channel(v)
                new_state_dict[f"{mod_name}.qweight_t"] = qweight.qweight_t.cpu()
                new_state_dict[f"{mod_name}.weight_scale"] = qweight.scale.cpu()
                scale_val = max(act_scales.get(mod_name, 1e-8), 1e-8) / 127.0
                new_state_dict[f"{mod_name}.static_act_scale"] = torch.tensor([scale_val], dtype=torch.float32).cpu()
                continue
        
        new_state_dict[k] = v.cpu()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ckpt.pt"
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt['model_state_dict'] = new_state_dict
    
    print(f"Saving quantized checkpoint to {out_path}...")
    torch.save(ckpt, out_path)
    print("Done!")

if __name__ == "__main__":
    main()
