#!/bin/bash
# B: score-precision sweep on the server.
#
# Asks whether pushing `item_scores` toward its measured ceiling (rho 0.99)
# makes the hierarchy competitive under learned dynamics.  D5 showed the score
# head is not the limit; the belief is.  So we sweep the two knobs that could
# move the belief: latent capacity and how hard it is pinned to the true state.
set -euo pipefail

source /opt/Anaconda3/etc/profile.d/conda.sh
conda activate WZLcosmos
cd /home/xys/code/WZL/next

echo "torch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo "gpu:   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python -u scripts/sweep_score_precision.py \
  --lt 3.0 --div 0.8 \
  --episodes 3000 --epochs 40 --holdout 200 \
  --latent-dims 64 128 256 --w-states 1.0 10.0 \
  --budget 512 --depth 2 --users 16 \
  --device cuda \
  --out results/score_precision.json
