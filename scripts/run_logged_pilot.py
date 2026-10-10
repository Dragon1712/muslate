"""E0/E1: strict exposed-feedback models versus a privileged synthetic upper bound.

python scripts/run_logged_pilot.py --out-dir results/e1_pilot --device cuda:0
Completed models and evaluations are immutable and reusable after interruption.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from muslate.config import EnvConfig, LogConfig, WorldModelConfig
from muslate.env.simulator import SlateSim
from muslate.env.logger import generate_log, generate_observed_log
from muslate.model.train import train_world_model, train_logged_world_model, get_device
from muslate.model.checkpoint import save_checkpoint, load_checkpoint
from muslate.model.neural_model import NeuralLatentModel
from muslate.eval.rollout import build_warm_start_cases, evaluate_from_prefixes, make_learned_latent_fn
from muslate.eval.calibration import spearman
from muslate.eval.integrity import validate_protocols
from muslate.planning.latent_model import CountingModel
from scripts.run_gate3 import build_variants


def build_pilot_arms(seeds=(1, 2, 3)) -> list[dict]:
    arms = []
    for seed in seeds:
        for condition, mode, weights in [('L0','none',[0]), ('L1','exposed',[1,10,100]),
                                          ('U1','oracle_full',[1,10,100])]:
            for weight in weights:
                arms.append(dict(id=f'{condition}_s{seed}_w{weight}', condition=condition,
                                 seed=seed, oracle_state=False, score_supervision=mode, w_score=weight))
    return arms


def evaluation_seed(seed: int, split: str) -> int:
    # Each run/split owns 1000 seeds; the 64-user population cannot cross a block.
    if split not in ('validation','test') or seed<0:
        raise ValueError('invalid evaluation seed/split')
    return 1000000+2000*seed+(1000 if split=='test' else 0)


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.json-')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def claim_run(root: Path, identity: dict) -> str:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != identity:
            raise ValueError(f'config/source identity conflict at {root}')
        result = root/'result.json'
        if result.exists():
            payload = json.loads(result.read_text(encoding='utf-8'))
            if payload.get('identity') != identity or payload.get('status') != 'complete':
                raise ValueError(f'invalid completed result at {root}')
            if not all((root/name).is_file() for name in ('model.pt','validation.json','test.json')):
                raise ValueError('completed run is missing checkpoint/evaluation dependencies')
            validate_protocols(payload.get('protocols',{}),identity['config'])
            model,metadata=load_checkpoint(root/'model.pt',torch.device('cpu'))
            validate_checkpoint(model,metadata,identity)
            if model.training_history!=payload.get('training_history'):
                raise ValueError('training history conflicts with checkpoint')
            for split in ('validation','test'):
                stage=json.loads((root/f'{split}.json').read_text(encoding='utf-8'))
                if (stage.get('identity')!=identity or stage.get('protocols')!=payload['protocols'][split]
                        or stage.get('diagnostics')!=payload['diagnostics'][split]):
                    raise ValueError('completed split dependency mismatch')
            return 'complete'
        return 'resume'
    if any(root.iterdir()):
        raise ValueError(f'nonempty directory without manifest: {root}')
    atomic_json(manifest, identity)
    return 'new'


@contextmanager
def run_lock(root: Path):
    """OS-owned lock: process death releases ownership, regardless of leftover file."""
    fd=os.open(root/'running.lock',os.O_CREAT|os.O_RDWR,0o600)
    with os.fdopen(fd,'r+',encoding='utf-8') as stream:
        if os.fstat(stream.fileno()).st_size==0:
            stream.write(' ');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'run already executing at {root}') from exc
        try:
            stream.seek(0)
            json.dump(dict(pid=os.getpid(),host=platform.node()),stream)
            stream.truncate();stream.flush()
            yield
        finally:
            stream.seek(0)
            if os.name=='nt': msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def validate_checkpoint(model, metadata, identity):
    settings=identity['config']
    env=settings['environment']
    n_items=env['catalog']['n_coarse']*env['catalog']['n_fine']
    if (metadata.get('identity')!=identity or asdict(model.cfg)!=settings['model']
            or model.n_items!=n_items or model.slate_size!=env['slate_size']):
        raise ValueError('checkpoint actual model/config identity conflict')


def source_fingerprint() -> dict:
    files = list((ROOT/'muslate').rglob('*.py'))
    files += [ROOT/'scripts'/name for name in ('run_logged_pilot.py','run_gate3.py')]
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def log_digest(log) -> str:
    digest = hashlib.sha256()
    for field in ('episode_ids','step_ids','slates','rewards','item_feedback'):
        array = getattr(log, field, None)
        if array is not None:
            digest.update(field.encode()); digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def runtime_info(device):
    return dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                cuda=torch.version.cuda, device=str(device), host=platform.node(),
                gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                cpu_threads=torch.get_num_threads())


def logged_diagnostics(nm, sim, log):
    episodes = len(log)//log.horizon
    slates = log.slates.reshape(episodes,log.horizon,log.slate_size)
    rewards = log.rewards.reshape(episodes,log.horizon)
    states = log.states.reshape(episodes,log.horizon,-1)
    latent = nm.encode_sequence(slates,rewards)
    predicted_scores = nm.item_scores(latent.reshape(-1,latent.shape[-1]))
    predicted_rewards = nm.reward(latent.reshape(-1,latent.shape[-1]),log.slates)
    rows=[]
    for index, scores in enumerate(predicted_scores):
        state = states.reshape(-1,states.shape[-1])[index]
        truth = sim.item_scores(state[:sim.cfg.dim],state[sim.cfg.dim:])
        rows.append(float(spearman(scores,truth)))
    error = predicted_rewards - log.rewards
    return dict(spearman_mean=float(np.mean(rows)), spearman_per_episode_step=rows,
                reward_mae=float(np.abs(error).mean()),reward_rmse=float(np.sqrt(np.square(error).mean())),
                data_digest=log_digest(log), seed_note='independent trajectories; shared user population')


def evaluate_protocol(nm, sim, args, split_seed, warmup, planner_seed):
    cases = build_warm_start_cases(sim,args.users,warmup,split_seed)
    counting = CountingModel(nm)
    planners = build_variants(counting,sim,args.budget,args.depth,planner_seed)
    rows={}
    for name, planner in planners.items():
        # Warm up a separate planner so timing excludes first-use kernels and
        # the evaluated planner's random stream remains untouched.
        warm_planner=build_variants(counting,sim,args.budget,args.depth,planner_seed)[name]
        first=cases[0]
        warm_latent=make_learned_latent_fn(nm)(first.state,first.hist_slates,first.hist_rewards)
        warm_planner.plan(warm_latent)
        if nm.device.type=='cuda': torch.cuda.synchronize(nm.device)
        counting.reset()
        ranks=[]
        def diagnostic(state, latent):
            ranks.append(float(spearman(nm.item_scores(latent)[0],sim.item_scores(state.h,state.sat))))
        start=time.perf_counter()
        result=evaluate_from_prefixes(sim,planner,make_learned_latent_fn(nm),cases,observer=diagnostic)
        rows[name]=dict(return_mean=result.mean,return_se=result.se,tail_mean=result.tail_mean,
                        tail_se=result.tail_se,returns=result.returns.tolist(),tail_returns=result.tail_returns.tolist(),
                        latency_ms=result.latency_ms,n_decisions=result.n_decisions,
                        slate_evals=counting.n_slate_evals,score_calls=counting.n_score_calls,
                        slate_evals_per_decision=counting.n_slate_evals/max(result.n_decisions,1),
                        closed_loop_spearman=float(np.mean(ranks)),closed_loop_spearman_per_user_step=ranks,
                        wall_seconds=time.perf_counter()-start)
        print(f'    {name}: return={result.mean:.4f}, tail={result.tail_mean:.4f}, '
              f'slate_evals/decision={rows[name]["slate_evals_per_decision"]:.1f}',flush=True)
    return dict(warmup=warmup,episode_seeds=[c.episode_seed for c in cases],
                user_ids=[c.user_id for c in cases],prefix_returns=[c.prefix_return for c in cases],
                planner_seed=planner_seed,rows=rows)


def run_arm(arm, args, source):
    device=get_device(args.device)
    cfg=WorldModelConfig(oracle_state=False,score_supervision=arm['score_supervision'],
                         w_score=arm['w_score'],seed=arm['seed'],epochs=args.epochs,
                         batch_size=args.batch_size,latent_dim=64,pairwise=True,n_ensemble=1)
    env_cfg=EnvConfig(long_term_strength=3.,div_penalty=.8,horizon=args.horizon)
    train_cfg=LogConfig(n_episodes=args.episodes,seed=10000+arm['seed'])
    settings=dict(arm=arm,model=asdict(cfg),environment=asdict(env_cfg),log=asdict(train_cfg),
                  holdout_episodes=args.holdout_episodes,users=args.users,warmup=args.warmup,
                  budget=args.budget,depth=args.depth,device=str(device),threads=args.threads,
                  validation_seed=20000+arm['seed'],test_seed=30000+arm['seed'],
                  validation_prefix_seed=evaluation_seed(arm['seed'],'validation'),
                  test_prefix_seed=evaluation_seed(arm['seed'],'test'),
                  planner_seed=60000+arm['seed'],protocol_version=1,
                  feedback='synthetic continuous unary response, noise std .02',
                  budget_note='nominal Gate3 proposal allocation; actual calls reported; no hard total budget claim')
    identity=dict(config=settings,source=source)
    root=Path(args.out_dir)/arm['id']
    status=claim_run(root,identity)
    if status=='complete':
        print(f'{arm["id"]}: complete, skipped',flush=True); return
    with run_lock(root):
        print(f'{arm["id"]}: {status} on {device}',flush=True)
        sim=SlateSim(env_cfg)
        checkpoint=root/'model.pt'
        if checkpoint.exists():
            model,metadata=load_checkpoint(checkpoint,device)
            validate_checkpoint(model,metadata,identity)
        else:
            # L0/L1 never build privileged logs. U1 is explicitly an upper bound.
            log=(generate_log(sim,train_cfg) if arm['condition']=='U1'
                 else generate_observed_log(sim,train_cfg))
            start=time.perf_counter()
            if arm['condition']=='U1': model=train_world_model(sim,log,cfg,device)
            else: model=train_logged_world_model(log,cfg,device)
            if device.type=='cuda': torch.cuda.synchronize(device)
            metadata=dict(identity=identity,train_seconds=time.perf_counter()-start,
                          train_data_digest=log_digest(log),runtime=runtime_info(device))
            save_checkpoint(model,checkpoint,metadata)
        nm=NeuralLatentModel([model],device)
        diagnostics={}; protocols={}
        for split,seed in [('validation',settings['validation_seed']),('test',settings['test_seed'])]:
            stage=root/f'{split}.json'
            if stage.exists():
                saved=json.loads(stage.read_text(encoding='utf-8'))
                if saved['identity']!=identity: raise ValueError('evaluation identity conflict')
                validate_protocols({split:saved.get('protocols',{})},settings,(split,))
            else:
                print(f'  {split} evaluation',flush=True)
                holdout=generate_log(sim,LogConfig(n_episodes=args.holdout_episodes,seed=seed))
                diag=logged_diagnostics(nm,sim,holdout)
                evaluations={}
                for protocol,warmup in [('full_episode',0),('common_prefix',args.warmup)]:
                    evaluations[protocol]=evaluate_protocol(nm,sim,args,settings[f'{split}_prefix_seed'],
                                                           warmup,settings['planner_seed'])
                saved=dict(identity=identity,diagnostics=diag,protocols=evaluations)
                atomic_json(stage,saved)
            diagnostics[split]=saved['diagnostics'];protocols[split]=saved['protocols']
        payload=dict(identity=identity,status='complete',metadata=metadata,training_history=model.training_history,
                     diagnostics=diagnostics,protocols=protocols,
                     completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        atomic_json(root/'result.json',payload)
        print(f'{arm["id"]}: COMPLETE',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--arm',default='all')
    ap.add_argument('--seeds',type=int,nargs='+',default=[1,2,3])
    ap.add_argument('--device',default='auto')
    ap.add_argument('--episodes',type=int,default=3000)
    ap.add_argument('--epochs',type=int,default=40)
    ap.add_argument('--users',type=int,default=16)
    ap.add_argument('--batch-size',type=int,default=512)
    ap.add_argument('--holdout-episodes',type=int,default=128)
    ap.add_argument('--horizon',type=int,default=20)
    ap.add_argument('--warmup',type=int,default=5)
    ap.add_argument('--budget',type=int,default=512)
    ap.add_argument('--depth',type=int,default=2)
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    if min(args.episodes,args.epochs,args.users,args.holdout_episodes,args.batch_size,args.threads)<=0:
        ap.error('counts must be positive')
    if not 0<=args.warmup<args.horizon or args.users>64 or args.budget<40 or args.depth<0:
        ap.error('invalid evaluation settings')
    if len(set(args.seeds))!=len(args.seeds): ap.error('duplicate seeds')
    if min(args.seeds)<0: ap.error('seeds must be nonnegative')
    arms=build_pilot_arms(args.seeds)
    if args.arm!='all': arms=[a for a in arms if a['id']==args.arm]
    if not arms: ap.error('unknown arm')
    torch.set_num_threads(args.threads)
    source=source_fingerprint()
    for arm in arms: run_arm(arm,args,source)


if __name__=='__main__': main()
