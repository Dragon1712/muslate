"""Freeze R2-C source/checkpoints/banks, then run two isolated GPU lanes."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from muslate.config import CatalogConfig,EnvConfig
from muslate.env.simulator import SlateSim
from muslate.model.checkpoint import load_checkpoint
from muslate.model.neural_model import NeuralLatentModel
from ops.r2c_diagnostics import make_bank,bank_hash,predict_case,summarize_case,aggregate


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def source_identity():
    files=list((ROOT/'muslate').rglob('*.py'))+list((ROOT/'ops').glob('r2c*.py'))
    files+=[ROOT/'tests/test_r2c.py',ROOT/'docs/research/2026-10-10-r2c-protocol.md']
    return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(files)}


def runtime(device=None):
    return dict(python=sys.version,executable=sys.executable,numpy=np.__version__,torch=torch.__version__,
                platform=platform.platform(),device=None if device is None else str(device),
                gpu=None if device is None or device.type!='cuda' else torch.cuda.get_device_name(device),
                cuda=torch.version.cuda,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                command=[sys.executable,*sys.argv],cwd=str(Path.cwd()),pid=os.getpid(),
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))


def checkpoint_path(root,condition,seed):
    return Path(root)/f'{condition}_s{seed}_w{0 if condition=="L0" else 100}'/'model.pt'


def validate_checkpoint(model,meta,condition,seed):
    weight=0 if condition=='L0' else 100;supervision='none' if condition=='L0' else 'exposed'
    arm=meta['identity']['config']['arm']
    expected=dict(id=f'{condition}_s{seed}_w{weight}',condition=condition,seed=seed,
                  w_score=weight,oracle_state=False,score_supervision=supervision)
    if any(arm.get(k)!=v for k,v in expected.items()): raise ValueError('checkpoint arm identity')
    if (model.cfg.seed!=seed or model.cfg.w_score!=weight or model.cfg.oracle_state
        or model.cfg.score_supervision!=supervision): raise ValueError('checkpoint config identity')


def simulator(settings):
    settings=copy.deepcopy(settings);settings['catalog']=CatalogConfig(**settings['catalog'])
    return SlateSim(EnvConfig(**settings))


def freeze(args):
    root=Path(args.out_dir)
    if root.exists(): raise FileExistsError('new output directory required')
    root.mkdir(parents=True)
    seeds=[1] if args.smoke else list(range(1,6));checkpoints={};environment=None
    for seed in seeds:
        for condition in ('L0','L1'):
            cp=checkpoint_path(args.checkpoint_root,condition,seed)
            model,meta=load_checkpoint(cp,torch.device('cpu'));validate_checkpoint(model,meta,condition,seed)
            current=meta['identity']['config']['environment']
            if environment is not None and environment!=current: raise ValueError('shared environment mismatch')
            environment=current
            checkpoints[f'{condition}_s{seed}']=dict(path=str(cp),sha256=digest(cp),training_identity=meta['identity'])
    sim=simulator(environment)
    if sim.cfg.horizon!=20: raise ValueError('original horizon20 required')
    manifest=dict(format_version=1,source_commit=args.source_commit,source=source_identity(),
                  protocol_sha256=digest(ROOT/'docs/research/2026-10-10-r2c-protocol.md'),
                  checkpoints=checkpoints,environment=environment,seeds=seeds,
                  expected_models=sorted(checkpoints),n_users=1 if args.smoke else 16,
                  n_sequences=64,sequence_length=5,probe_count=16,horizons=[1,3,5],
                  seed_to_gpu={str(s):3 if s%2 else 4 for s in seeds},smoke=args.smoke,runtime=runtime())
    # The protocol snapshot is published before any bank, and banks before predictions.
    atomic(root/'protocol.json',manifest);banks={}
    for seed in seeds:
        print(f'BANK seed={seed}',flush=True)
        bank=make_bank(sim,seed,manifest['n_users'])
        path=root/'banks'/f'seed{seed}.json';atomic(path,bank)
        banks[str(seed)]=dict(path=path.relative_to(root).as_posix(),sha256=digest(path),content_sha256=bank_hash(bank))
    atomic(root/'bank_manifest.json',dict(protocol_sha256=digest(root/'protocol.json'),banks=banks,
                                        frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    return manifest


def run_seed(args):
    root=Path(args.out_dir);protocol=json.loads((root/'protocol.json').read_text());seed=args.seed
    if protocol['source_commit']!=args.source_commit or protocol['source']!=source_identity():
        raise ValueError('source freeze mismatch')
    if seed not in protocol['seeds']: raise ValueError('seed outside protocol')
    frozen=json.loads((root/'bank_manifest.json').read_text());bank_entry=frozen['banks'][str(seed)]
    bankpath=root/bank_entry['path']
    if frozen['protocol_sha256']!=digest(root/'protocol.json') or bank_entry['sha256']!=digest(bankpath):
        raise ValueError('bank/protocol freeze mismatch')
    bank=json.loads(bankpath.read_text())
    if bank_hash(bank)!=bank_entry['content_sha256']: raise ValueError('bank content freeze mismatch')
    device=torch.device(args.device);torch.set_num_threads(4)
    if str(device)!=f'cuda:{protocol["seed_to_gpu"][str(seed)]}': raise ValueError('reserved GPU mismatch')
    for condition in ('L0','L1'):
        label=f'{condition}_s{seed}';entry=protocol['checkpoints'][label];cp=Path(entry['path'])
        if digest(cp)!=entry['sha256']: raise ValueError('checkpoint changed')
        model,meta=load_checkpoint(cp,device);validate_checkpoint(model,meta,condition,seed)
        if meta['identity']!=entry['training_identity']: raise ValueError('checkpoint metadata changed')
        nm=NeuralLatentModel([model],device);rows=[];start=time.perf_counter()
        for case in bank['cases']:
            print(f'PREDICT {label} user={case["user_id"]}',flush=True)
            pred=predict_case(nm,case)
            rows.append(dict(user_id=case['user_id'],initial_state_sha256=case['initial_state_sha256'],
                             history_sha256=case['history_sha256'],prediction={k:v.tolist() for k,v in pred.items()},
                             metrics=summarize_case(case,pred,protocol['environment']['gamma'])))
        torch.cuda.synchronize(device)
        row=dict(status='complete',condition=condition,seed=seed,source_commit=args.source_commit,
                 identity=dict(source=protocol['source'],protocol_file_sha256=digest(root/'protocol.json'),
                               checkpoint=entry,bank=bank_entry,training_identity=meta['identity']),
                 runtime=runtime(device),cases=rows,aggregate=aggregate(rows),wall_seconds=time.perf_counter()-start)
        atomic(root/'models'/f'{label}.json',row)
        print(f'DONE {label} seconds={row["wall_seconds"]:.2f}',flush=True)
    atomic(root/f'seed{seed}_complete.json',dict(status='complete',seed=seed))


def queue(args):
    protocol=freeze(args);root=Path(args.out_dir)
    def lane(gpu):
        jobs=[]
        for seed in protocol['seeds']:
            if protocol['seed_to_gpu'][str(seed)]!=gpu: continue
            command=[sys.executable,str(ROOT/'ops/r2c_run.py'),'--out-dir',str(root),
                     '--checkpoint-root',args.checkpoint_root,'--source-commit',args.source_commit,
                     '--seed',str(seed),'--device',f'cuda:{gpu}']
            with (root/f'seed{seed}.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                                      env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
            jobs.append(dict(seed=seed,gpu=gpu,command=command,exit_code=result.returncode))
        return jobs
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[job for lane_jobs in pool.map(lane,[3,4]) for job in lane_jobs]
    present=sorted(p.stem for p in (root/'models').glob('*.json'))
    missing=sorted(set(protocol['expected_models'])-set(present))
    atomic(root/'queue_status.json',dict(jobs=jobs,present=present,missing=missing,
                                       finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    if missing or any(j['exit_code'] for j in jobs): raise RuntimeError('incomplete queue')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',required=True);p.add_argument('--checkpoint-root',required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--seed',type=int);p.add_argument('--device',default='cuda:3');args=p.parse_args()
    if len(args.source_commit)!=40 or any(c not in '0123456789abcdef' for c in args.source_commit):
        p.error('--source-commit must be a full hexadecimal commit SHA')
    try: run_seed(args) if args.seed is not None else queue(args)
    except Exception as error:
        root=Path(args.out_dir)
        # Never overwrite an existing failure or artifact.
        if root.exists():
            atomic(root/f'failure_{args.seed if args.seed is not None else "queue"}.json',
                   dict(error=repr(error),traceback=traceback.format_exc(),runtime=runtime()))
        raise


if __name__=='__main__': main()
