import csv
import time
import torch
from contextlib import nullcontext

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

