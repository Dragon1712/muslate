#!/usr/bin/env bash
# Install torch into the WZLnext env, in the background, logging to /tmp.
# Uses the aliyun mirror (the tsinghua mirror is unreachable from this host).
set -uo pipefail

PY=/home/xys/.conda/envs/WZLnext/bin/python
PIP=/home/xys/.conda/envs/WZLnext/bin/pip
LOG=/tmp/wzlnext_pip.log

echo "python : $($PY -V 2>&1)"
echo "pip    : $($PIP --version 2>&1)"

if "$PY" -c "import torch" >/dev/null 2>&1; then
    echo "torch already importable: $("$PY" -c 'import torch; print(torch.__version__)')"
    exit 0
fi

echo "starting background install -> $LOG"
nohup "$PIP" install --index-url https://mirrors.aliyun.com/pypi/simple/ \
      torch numpy > "$LOG" 2>&1 &
echo "pid=$!"
echo
echo "poll with:  tail -5 $LOG ; $PY -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
