"""Fresh output only; three GPU slots and serialized seeds within each slot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.r2b_run import atomic,digest,source_identity,ROOT,PROTOCOL
from ops.r2b_search import METHODS
from ops.r2b_identity import require_commit,runtime_context,GPU_SLOTS,gpu_for_seed

def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',required=True);p.add_argument('--checkpoint-root',required=True)
    p.add_argument('--calibration',required=True);p.add_argument('--source-commit',required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    root=Path(args.out_dir)
    require_commit(args.source_commit)
    if root.exists():raise ValueError('fresh output directory required')
    calibration=json.loads(Path(args.calibration).read_text())
    if calibration['source']!=source_identity():raise ValueError('calibration source differs from frozen runner')
    seeds=[1] if args.smoke else list(range(1,6));users=1 if args.smoke else 16
    expected=[f'L1_s{s}_{m}' for s in seeds for m in METHODS]
    gpu_by_seed={s:gpu_for_seed(s) for s in seeds};root.mkdir(parents=True)
    atomic(root/'manifest.json',dict(source=source_identity(),source_commit=args.source_commit,
        protocol_sha256=digest(ROOT/PROTOCOL),calibration_sha256=digest(args.calibration),
        calibration_path=str(Path(args.calibration).resolve()),expected_cells=expected,expected_diagnostics=[f'L1_s{s}' for s in seeds],
        checkpoint_root=args.checkpoint_root,queue_runtime=runtime_context(),out_dir_argument=args.out_dir,calibration_argument=args.calibration,
        seeds=seeds,users=users,seed_to_gpu=gpu_by_seed,smoke=args.smoke,prefix_length=5,horizon=20,scope='single exploratory mid estimated-compute budget',
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    atomic(root/'calibration.json',json.loads(Path(args.calibration).read_text()))
    def slot(gpu):
        results=[]
        for seed in [s for s in seeds if gpu_by_seed[s]==gpu]:
            cmd=[sys.executable,'ops/r2b_run.py','--out-dir',str(root),'--checkpoint-root',args.checkpoint_root,
                '--calibration',args.calibration,'--source-commit',args.source_commit,'--seed',str(seed),
                '--device',f'cuda:{gpu}','--users',str(users)]
            with (root/f'seed{seed}.log').open('w') as stream:
                result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
            results.append(dict(seed=seed,gpu=gpu,exit_code=result.returncode,command=cmd))
        return results
    with ThreadPoolExecutor(max_workers=len(GPU_SLOTS)) as pool:jobs=[r for batch in pool.map(slot,GPU_SLOTS) for r in batch]
    complete=[p.stem for p in (root/'cells').glob('*.json')];probes=[p.stem for p in (root/'diagnostics').glob('*.json')]
    missing=sorted(set(expected)-set(complete));missing_probes=sorted({f'L1_s{s}' for s in seeds}-set(probes))
    atomic(root/'queue_status.json',dict(jobs=jobs,complete=complete,missing=missing,missing_diagnostics=missing_probes,
       finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    if missing or missing_probes or any(j['exit_code'] for j in jobs):raise SystemExit(1)
if __name__=='__main__':main()
