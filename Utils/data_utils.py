import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


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
