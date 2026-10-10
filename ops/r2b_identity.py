"""Exact launch command and observable runtime metadata; no inferred model identity."""
import os
import json
from pathlib import Path
import platform
import re
import socket
import sys
import time
import subprocess

GPU_SLOTS=(2,4,5)

def gpu_for_seed(seed):
    if seed not in range(1,6):raise ValueError('training seed outside frozen scope')
    return GPU_SLOTS[(seed-1)%len(GPU_SLOTS)]

def resource_snapshot():
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],
                              capture_output=True,text=True,timeout=10)
        gpus=result.stdout.strip() if result.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):gpus=None
    return dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                host_load_average=list(os.getloadavg()) if hasattr(os,'getloadavg') else None,gpu_snapshot=gpus)

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    temp=path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    try:
        with temp.open('x',encoding='utf-8') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False))
        os.link(temp,path)
    finally:
        if temp.exists():temp.unlink()

def require_commit(commit):
    if not isinstance(commit,str) or re.fullmatch('[0-9a-f]{40}',commit) is None:
        raise ValueError('full 40-hex source commit required')

def runtime_context(versions=None):
    return dict(command=[sys.executable]+sys.argv,python=sys.version,executable=sys.executable,
                cwd=str(Path.cwd()),hostname=socket.gethostname(),pid=os.getpid(),platform=platform.platform(),
                libraries=versions or {})

def validate_command(command,expected):
    if not isinstance(command,list) or len(command)<2:raise ValueError('missing actual command')
    for flag,value in expected.items():
        if command.count(flag)!=1:raise ValueError('missing/duplicated command flag')
        index=command.index(flag)
        if index+1>=len(command) or command[index+1]!=str(value):raise ValueError('command flag mismatch: '+flag)
    require_commit(command[command.index('--source-commit')+1])

def validate_runtime(runtime,expected):
    validate_command(runtime['command'],expected)
    for field in ('python','executable','cwd','hostname','pid','platform','libraries'):
        if field not in runtime or runtime[field] in (None,''):raise ValueError('runtime metadata missing')
