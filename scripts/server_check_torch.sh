#!/usr/bin/env bash
# Report whether the WZLnext env can import torch, and whether it sees the GPUs.
set -uo pipefail

PY="${PY:-/home/xys/.conda/envs/WZLnext/bin/python}"

if ! "$PY" -c 'import torch' 2>/dev/null; then
    echo "TORCH_MISSING"
    exit 1
fi

"$PY" - <<'PY'
import torch
print("TORCH_OK", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
if torch.cuda.is_available():
    print("  arch list:", [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")])
    for i in range(torch.cuda.device_count()):
        print(f"  gpu{i}:", torch.cuda.get_device_name(i))
PY
