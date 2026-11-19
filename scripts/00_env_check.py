"""Phase 0: record the machine, the Python environment and the storage budget."""
import os
import platform
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common import ROOT, SEED, set_seed, save_json


def folder_gb(p):
    tot = 0
    for dp, _, fs in os.walk(p):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return tot / 1e9


def main():
    set_seed(SEED)
    import torch
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        gpu = ""
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_cores_logical": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "physical_gpu_detected": gpu or None,
        "free_disk_gb_on_project_drive": round(shutil.disk_usage(ROOT).free / 1e9, 1),
        "seed": SEED,
        "storage_budget_gb": 6.0,
        "project_folder_gb_now": round(folder_gb(ROOT), 3),
        "decision": ("A GPU is physically present but the installed torch is the CPU-only build: the "
                     "CUDA build of torch plus its NVIDIA libraries needs roughly 4-5 GB, which would "
                     "consume most of the 6 GB budget. Everything is therefore designed for CPU "
                     "(frozen backbone + linear probe); end-to-end fine-tuning is skipped."),
    }
    p = save_json(out, "phase0")
    print(open(p).read())


if __name__ == "__main__":
    main()
