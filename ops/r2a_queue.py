"""One seed per GPU; serial measurements within each GPU, isolated R2 jobs only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.run_r2a import atomic,source_identity,digest,tiny_diagnostic,ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',required=True);p.add_argument('--checkpoint-root',required=True)
    p.add_argument('--smoke',action='store_true');args=p.parse_args();root=Path(args.out_dir)
    if root.exists():raise ValueError('new output directory required')
    root.mkdir(parents=True)
    seeds=[1] if args.smoke else list(range(1,6))
    expected=[f'{c}_s{s}_{p}_d{d}' for s in seeds for c in ('L0','L1') for p in ('flat_uniform','sid_hier') for d in range(4)]
    expected_probes=[] if args.smoke else [f'{c}_s{s}_{p}' for s in seeds for c in ('L0','L1') for p in ('flat_uniform','sid_hier')]
    atomic(root/'protocol.json',dict(source=source_identity(),protocol_sha256=digest(ROOT/'docs/research/2026-10-09-r2a-protocol.md'),
           expected_cells=expected,expected_probes=expected_probes,seed_to_gpu={str(s):s-1 for s in seeds},users=1 if args.smoke else 16,
           probe_users=0 if args.smoke else 4,scope='exploratory, fixed nominal B128, no equal-compute claim',
           started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    atomic(root/'tiny_oracle.json',tiny_diagnostic())
    def worker(seed):
        cmd=[sys.executable,'ops/run_r2a.py','--out-dir',str(root),'--checkpoint-root',args.checkpoint_root,
             '--seed',str(seed),'--device',f'cuda:{seed-1}','--users','1' if args.smoke else '16']
        if not args.smoke:cmd.append('--probes')
        with (root/f'seed{seed}.log').open('w') as stream:
            result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
        return dict(seed=seed,exit_code=result.returncode)
    with ThreadPoolExecutor(max_workers=len(seeds)) as pool:results=list(pool.map(worker,seeds))
    complete=[p.stem for p in (root/'cells').glob('*.json')]
    atomic(root/'queue_status.json',dict(jobs=results,complete=complete,missing=sorted(set(expected)-set(complete)),
          finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    if any(r['exit_code'] for r in results):raise SystemExit(1)


if __name__=='__main__':main()
