#!/usr/bin/env bash
# Server-side one-shot runner for the large-scale validation.
#
# Self-checks first, then fans the experiments out across every visible GPU.
# Each job gets its own CUDA_VISIBLE_DEVICES and writes its own log, so a single
# failed job does not take the run down.
#
# Usage (after `conda activate WZLnext`):
#     bash scripts/server_run_all.sh
#     EPISODES=40000 EPOCHS=100 bash scripts/server_run_all.sh
#
# Outputs land in results/server/ (one .json + .md per experiment, plus logs).

set -uo pipefail

# Activate the project's conda env if we are not already inside one, so the
# script works from a bare non-login shell (e.g. `ssh host 'bash script.sh'`).
if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    for base in /opt/Anaconda3 /opt/miniconda3 "$HOME/miniconda3" "$HOME/anaconda3"; do
        if [ -f "$base/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$base/etc/profile.d/conda.sh"
            conda activate "${CONDA_ENV:-WZLnext}" && break
        fi
    done
fi
echo "conda env: ${CONDA_DEFAULT_ENV:-<none>}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EPISODES="${EPISODES:-20000}"
EPOCHS="${EPOCHS:-60}"
ENSEMBLE="${ENSEMBLE:-4}"
ERR_EPISODES="${ERR_EPISODES:-5000}"
ERR_EPOCHS="${ERR_EPOCHS:-150}"
USERS="${USERS:-32}"
CANDIDATES="${CANDIDATES:-32}"
LTS="${LTS:-3.0 5.0}"
LT_GATE4="${LT_GATE4:-3.0}"
OUT="results/server"
mkdir -p "$OUT"

echo "=============================================================="
echo " MuSlate server run"
echo "   episodes/member : $EPISODES"
echo "   epochs          : $EPOCHS"
echo "   ensemble        : $ENSEMBLE"
echo "   error-model data: ${ERR_EPISODES} x 2"
echo "   users           : $USERS"
echo "=============================================================="

echo
echo "== environment =="
python - <<'PY'
import sys, platform
print("python  :", sys.version.split()[0], platform.machine())
try:
    import torch
    print("torch   :", torch.__version__, "| cuda:", torch.cuda.is_available(),
          "| gpus:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  gpu{i}  :", torch.cuda.get_device_name(i))
except Exception as exc:                                   # noqa: BLE001
    print("torch   : UNAVAILABLE ->", exc)
    sys.exit(1)
import numpy
print("numpy   :", numpy.__version__)
PY
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

echo
echo "== self-check (tests) =="
FAILED=0
for t in env planning world_model conservative error_model; do
    if python "tests/test_$t.py" > "$OUT/test_$t.log" 2>&1; then
        echo "  ok    test_$t  ($(tail -1 "$OUT/test_$t.log"))"
    else
        echo "  FAIL  test_$t   -- see $OUT/test_$t.log"
        FAILED=1
    fi
done
if [ "$FAILED" -ne 0 ]; then
    echo
    echo "Self-check failed. Not starting the large runs."
    exit 1
fi

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo
    echo "CHECK_ONLY=1 -> environment and tests are good; stopping before the runs."
    exit 0
fi

NGPU=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")
echo
echo "== launching experiments on $NGPU GPU(s) =="

PIDS=()
# Round-robin across GPUs.  This deliberately does NOT use "$(gpu_for)" command
# substitution: that runs in a subshell, so the slot counter would never advance
# and every job would silently land on GPU 0.
GPU_SLOT=0
launch() {  # launch <name> <cmd...>
    local name="$1"; shift
    local gpu=$(( GPU_SLOT % NGPU ))
    GPU_SLOT=$(( GPU_SLOT + 1 ))
    echo "  [gpu $gpu] $name"
    CUDA_VISIBLE_DEVICES="$gpu" "$@" > "$OUT/$name.log" 2>&1 &
    PIDS+=("$!")
}

# 1. Calibration: is the error model still calibrated at scale?
launch calibration \
    python scripts/run_calibration.py \
    --episodes "$EPISODES" --epochs "$EPOCHS" --ensemble "$ENSEMBLE" \
    --err-episodes "$ERR_EPISODES" --err-epochs "$ERR_EPOCHS" \
    --eval-episodes 500 --out "$OUT/calibration.json"

# 2. The decisive end-to-end run: does the penalty restore multi-step value?
launch gate4_error_model \
    python scripts/run_gate4.py \
    --episodes "$EPISODES" --epochs "$EPOCHS" --ensemble 2 \
    --unc-source error_model --err-episodes "$ERR_EPISODES" --err-epochs "$ERR_EPOCHS" \
    --users "$USERS" --candidates "$CANDIDATES" --horizons 0 1 2 3 5 \
    --warmup 5 --lt "$LT_GATE4" --out "$OUT/gate4_error_model.json"

# 3. Control: the same run with ensemble disagreement instead of the error model
launch gate4_ensemble \
    python scripts/run_gate4.py \
    --episodes "$EPISODES" --epochs "$EPOCHS" --ensemble 2 \
    --unc-source ensemble \
    --users "$USERS" --candidates "$CANDIDATES" --horizons 0 1 2 3 5 \
    --warmup 5 --lt "$LT_GATE4" --out "$OUT/gate4_ensemble.json"

# 4. Gate 2 at scale: multi-step value vs long-horizon coupling
launch gate2 \
    python scripts/run_gate2.py --dynamics both \
    --lts 0 0.5 1 2 3 5 --horizons 0 1 2 3 5 \
    --users "$USERS" --candidates "$CANDIDATES" --beam 3 --warmup 5 \
    --episodes "$EPISODES" --epochs "$EPOCHS" --out "$OUT/gate2.json"

# 5. Gate 3 at scale: action abstraction at equal budget
launch gate3 \
    python scripts/run_gate3.py --dynamics both \
    --lts 0 1 3 5 --div 0.8 --budget 512 --depth 2 \
    --users "$USERS" --episodes "$EPISODES" --epochs "$EPOCHS" \
    --out "$OUT/gate3.json"

# 6. Q3 at scale: does the tree still win under combinatorial reward?
launch q3 \
    python scripts/run_q3.py --dynamics oracle \
    --lts 3.0 --div 0.8 --depth 2 --users "$USERS" \
    --candidates "$CANDIDATES" --samples "$CANDIDATES" --n-sims 32 \
    --out "$OUT/q3.json"

echo
echo "waiting for ${#PIDS[@]} job(s) ..."
STATUS=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
done

echo
echo "=============================================================="
echo " results"
echo "=============================================================="
for md in "$OUT"/*.md; do
    [ -e "$md" ] || continue
    echo
    cat "$md"
done

echo
if [ "$STATUS" -eq 0 ]; then
    echo "ALL JOBS COMPLETED. Logs in $OUT/"
else
    echo "SOME JOBS FAILED. Check the per-job logs in $OUT/:"
    ls -1 "$OUT"/*.log 2>/dev/null | sed 's/^/  /'
fi
