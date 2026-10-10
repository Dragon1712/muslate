#!/bin/bash
# B2 control: can the belief be sharpened WITHOUT privileged state access?
#
# B (oracle_state=True) showed `hier - flat` turning positive (+0.94..+1.22) once
# the steady-state D2 rose from 0.84 to 0.97, and that the decisive knob was the
# state-supervision weight, not latent capacity.  But that arm leans on ground
# truth state, so it cannot yet support a claim about a real system.
#
# Here the state target is removed.  The only direct handle left on the belief is
# the item-score loss, so that is what gets swept.  If D2 and `hier - flat` still
# recover, the B conclusion survives without the oracle; if they do not, the
# recovery was an artefact of privileged supervision and must be reported as such.
set -euo pipefail

source /opt/Anaconda3/etc/profile.d/conda.sh
conda activate WZLcosmos
cd /home/xys/code/WZL/next

python -u scripts/sweep_score_precision.py \
  --lt 3.0 --div 0.8 \
  --episodes 3000 --epochs 40 --holdout 200 \
  --latent-dims 64 128 256 \
  --no-oracle-state --w-scores 1.0 10.0 100.0 \
  --budget 512 --depth 2 --users 16 \
  --device cuda \
  --out results/score_precision_no_oracle.json
