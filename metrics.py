import math
from sklearn.metrics import roc_auc_score

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

