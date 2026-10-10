"""Calibration, frozen R2-B seed cells, and fixed-state diagnostics."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
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
from muslate.eval.rollout import build_warm_start_cases,make_learned_latent_fn
from ops.r2b_search import Search,METHODS,RANDOM_SEEDS,labels_for,generate,prepare
from ops.r2b_identity import runtime_context,require_commit,atomic,resource_snapshot

PROTOCOL='docs/research/2026-10-10-r2b-protocol.md'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_identity():
    files=list((ROOT/'muslate').rglob('*.py'))+list((ROOT/'ops').glob('r2b*.py'))+[ROOT/PROTOCOL,ROOT/'tests/test_r2b.py']
    return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(files)}
def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)
def load(root,seed,device):
    checkpoint=Path(root)/f'L1_s{seed}_w100/model.pt'; model,meta=load_checkpoint(checkpoint,device)
    arm=meta['identity']['config']['arm']
    expected=dict(id=f'L1_s{seed}_w100',condition='L1',seed=seed,w_score=100,oracle_state=False,score_supervision='exposed')
    if any(arm.get(k)!=v for k,v in expected.items()) or model.cfg.seed!=seed or model.cfg.w_score!=100 or model.cfg.oracle_state or model.cfg.score_supervision!='exposed':
        raise ValueError('checkpoint identity conflict')
    settings=copy.deepcopy(meta['identity']['config']['environment']);settings['catalog']=CatalogConfig(**settings['catalog'])
    sim=SlateSim(EnvConfig(**settings))
    if sim.cfg.horizon!=20:raise ValueError('20-step environment required')
    nm=NeuralLatentModel([model],device)
    return nm,sim,dict(checkpoint_sha256=digest(checkpoint),checkpoint_path=str(checkpoint),training_identity=meta['identity'],
                       environment=asdict(sim.cfg),runtime=dict(**runtime_context(dict(torch=torch.__version__,numpy=np.__version__)),
                       device=str(device),gpu=torch.cuda.get_device_name(device)))
def method_labels(reference,method):
    return labels_for(reference,int(method.split('_')[-1])) if method.startswith('random') else labels_for(reference)
def decision_key(seed,user,t):return 31000000+100000*seed+100*user+t

def timing(fn,device,repetitions=15):
    for _ in range(5):fn()
    raw=[]
    for _ in range(repetitions):
        sync(device);start=time.perf_counter();fn();sync(device);raw.append(1000*(time.perf_counter()-start))
    return raw

def calibrate(args):
    resources_before=resource_snapshot()
    device=torch.device(args.device);torch.set_num_threads(4);raw={};state_records=[];identities=[]
    all_reward={str(n):[] for n in range(1,33)};score=[];cpu={k:[] for k in ('setup','uniform','sid','refine')};states=[]
    for seed in (1,2):
        nm,sim,identity=load(args.checkpoint_root,seed,device);identities.append(identity)
        cases=build_warm_start_cases(sim,10,5,32000000+2000*seed);latent_fn=make_learned_latent_fn(nm)
        latent=latent_fn(cases[0].state,cases[0].hist_slates,cases[0].hist_rewards)
        labels=sim.catalog.item_coarse;scores=nm.item_scores(latent)[0];order,groups,p=prepare(scores,labels)
        rng=np.random.default_rng(222+seed)
        slates=np.stack([generate(rng,nm.n_items,nm.slate_size) for _ in range(32)])
        for n in range(1,33):
            all_reward[str(n)]+=timing(lambda n=n:nm.reward(np.repeat(latent,n,axis=0),slates[:n]),device)
        score+=timing(lambda:nm.item_scores(latent),device)
        # Include construction, stable tuple/dedup/cache checking and best-value bookkeeping;
        # these are estimated CPU proxies, not an exact decomposition of deployed latency.
        def overhead(kind):
            if kind=='setup':prepare(scores,labels);return
            seen={};best=-np.inf
            offset=int(rng.integers(max(1,(len(order)+31)//32)))*32
            current=order[:nm.slate_size]
            for j in range(32):
                if kind in ('uniform','sid'):
                    action=generate(rng,nm.n_items,nm.slate_size,groups if kind=='sid' else None,p if kind=='sid' else None)
                else:
                    # Same O(1) ranked-slot legality check / slate copy as Search.refine,
                    # rather than repeatedly building a full-catalog legality list.
                    item=int(order[(offset+j)%len(order)]);pos=0
                    if item==int(current[pos]) or item in current:continue
                    action=current.copy();action[pos]=item
                key=tuple(map(int,action));seen[key]=float(j);best=max(best,seen[key])
        for kind in cpu:cpu[kind]+=timing(lambda kind=kind:overhead(kind),device)
        for case in cases:
            state=latent_fn(case.state,case.hist_slates,case.hist_rewards)
            states.append((nm,state,labels,seed,case.user_id))
            state_records.append(dict(seed=seed,user=case.user_id,episode_seed=case.episode_seed,
                                      slates=[a.tolist() for a in case.hist_slates],rewards=case.hist_rewards))
    medians={str(n):float(np.median(all_reward[str(n)])) for n in range(1,33)}
    # Monotone envelope is essential: a deduplicated smaller batch must not cost more
    # than the worst-case batch preflight reservation.
    envelope=np.maximum.accumulate([medians[str(n)] for n in range(1,33)])
    coefficients=dict(reward_ms={str(n):float(envelope[n-1]) for n in range(1,33)},
        score_ms=float(np.median(score)),setup_ms=float(np.median(cpu['setup'])),
        **{kind+'_ms':float(np.median(cpu[kind])/32) for kind in ('uniform','sid','refine')},max_batch=32)
    references=[]
    for nm,state,labels,seed,user in states:
        _,row=Search(nm,coefficients,1e6,32000000+100000*seed+user,labels).plan(state,'true_sid_hier',reference=True)
        if row['error'] or row['stop_reason']!='reference_complete':raise ValueError('calibration reference failure')
        references.append(dict(seed=seed,user=user,estimated_ms=row['estimated_ms'],cost=row['cost']))
    cap=float(np.median([r['estimated_ms'] for r in references]));minimum=coefficients['score_ms']+coefficients['setup_ms']+coefficients['reward_ms']['1']
    if cap<minimum:raise ValueError('reference median cannot fit common anchor')
    fit=np.polyfit(np.arange(1,33),list(medians.values()),1)
    result=dict(coefficients=coefficients,mid_cap_ms=cap,minimum_ms=minimum,reward_raw_ms=all_reward,score_raw_ms=score,
        proposal_raw_batch32_ms=cpu,reward_raw_medians_ms=medians,
        reward_linear_fit=dict(call_overhead_ms=float(fit[1]),sample_ms=float(fit[0]),
              residuals_ms=[float(medians[str(n)]-(fit[1]+fit[0]*n)) for n in range(1,33)]),
        reference_states=references,calibration_states=state_records,checkpoints=identities,source=source_identity(),
        limitations='Timing proxies; calibrated GPU2 lookup is shared across GPU2/4/5. Other GPUs may host unrelated jobs; CPU contention is observable but uncontrolled. Estimated cap is not wall-clock or FLOPs.',
        runtime=runtime_context(dict(torch=torch.__version__,numpy=np.__version__)),
        resources_before=resources_before,resources_after=resource_snapshot(),
        device=str(device),created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    if Path(args.out_dir).exists():raise ValueError('fresh calibration path required')
    atomic(Path(args.out_dir)/'calibration.json',result);print(json.dumps(dict(cap_ms=cap,minimum_ms=minimum)),flush=True)

def run_cell(nm,sim,cases,method,seed,calibration,identity):
    latent_fn=make_learned_latent_fn(nm);decisions=[];returns=[];tails=[]
    labels=method_labels(sim.catalog.item_coarse,method)
    for case in cases:
        state=case.state.clone();hs=[a.copy() for a in case.hist_slates];hr=list(case.hist_rewards);tail=0.
        while state.t<sim.cfg.horizon:
            sync(nm.device);start=time.perf_counter();latent=latent_fn(state,hs,hr);sync(nm.device);encoding=1000*(time.perf_counter()-start)
            start=time.perf_counter();key=decision_key(seed,case.user_id,state.t)
            action,row=Search(nm,calibration['coefficients'],calibration['mid_cap_ms'],key,labels).plan(latent,method)
            sync(nm.device);latency=1000*(time.perf_counter()-start);t=state.t
            reward,state=sim.step(state,action,noise=False);tail+=sim.cfg.gamma**t*reward
            decisions.append(dict(user_id=case.user_id,t=t,decision_key=key,slate=action.tolist(),reward=float(reward),
                planning=row,latency_ms=latency,encoding_ms=encoding,overestimation=row['predicted_reward']-float(reward)))
            hs.append(action);hr.append(float(reward))
        tails.append(float(tail));returns.append(float(case.prefix_return+tail))
    return dict(status='complete',identity=identity,method=method,seed=seed,user_ids=[c.user_id for c in cases],
        episode_seeds=[c.episode_seed for c in cases],prefix_returns=[c.prefix_return for c in cases],
        prefix_actions=[[a.tolist() for a in c.hist_slates] for c in cases],prefix_rewards=[c.hist_rewards for c in cases],
        returns=returns,tail_returns=tails,return_mean=float(np.mean(returns)),decisions=decisions)

def diagnostics(nm,sim,cases,seed,calibration,identity):
    latent_fn=make_learned_latent_fn(nm);rows=[]
    for case in cases[:4]:
        latent=latent_fn(case.state,case.hist_slates,case.hist_rewards);trials=[]
        for method in ('flat_coordinate_refine','true_sid_hier')+tuple(f'random_sid_hier_{s}' for s in RANDOM_SEEDS):
            start=time.perf_counter()
            action,row=Search(nm,calibration['coefficients'],calibration['mid_cap_ms'],decision_key(seed,case.user_id,5),
                              method_labels(sim.catalog.item_coarse,method)).plan(latent,method,diagnostic=True)
            sync(nm.device);latency=1000*(time.perf_counter()-start)
            r,_=sim.step(case.state.clone(),action,noise=False);anchor_r,_=sim.step(case.state.clone(),np.array(row['anchor']),noise=False)
            predicted=row['predicted_reward']-row['anchor_prediction'];truth=float(r-anchor_r)
            trials.append(dict(method=method,slate=action.tolist(),planning=row,latency_ms=latency,
                               true_reward=float(r),anchor_true_reward=float(anchor_r),predicted_improvement=predicted,
                               true_improvement=truth,selection_bias=predicted-truth))
        rows.append(dict(user_id=case.user_id,trials=trials))
    return dict(identity=identity,seed=seed,rows=rows)

def run(args):
    require_commit(args.source_commit)
    root=Path(args.out_dir);manifest=json.loads((root/'manifest.json').read_text());calibration=json.loads(Path(args.calibration).read_text())
    if list(root.glob(f'failure_s{args.seed}*.json')) or list((root/'cells').glob(f'L1_s{args.seed}_*.json')):
        raise ValueError('standalone retry/partial resume forbidden')
    source=source_identity()
    if manifest['source']!=source or manifest['source_commit']!=args.source_commit or manifest['calibration_sha256']!=digest(args.calibration):
        raise ValueError('source/calibration freeze conflict')
    device=torch.device(args.device);torch.set_num_threads(4);nm,sim,identity=load(args.checkpoint_root,args.seed,device)
    identity.update(source=source,source_commit=args.source_commit,protocol_sha256=digest(ROOT/PROTOCOL),
                    calibration_sha256=digest(args.calibration),prefix_seed=31000000+2000*args.seed)
    cases=build_warm_start_cases(sim,args.users,5,identity['prefix_seed'])
    # Warmup outside measured decisions; no true-return tuning.
    latent=make_learned_latent_fn(nm)(cases[0].state,cases[0].hist_slates,cases[0].hist_rewards)
    for _ in range(5):nm.reward(np.repeat(latent,32,axis=0),np.tile(np.arange(nm.slate_size),(32,1)))
    for method in METHODS:
        path=root/'cells'/f'L1_s{args.seed}_{method}.json'
        if path.exists():raise ValueError('cell overwrite forbidden')
        print('START',path.stem,flush=True);row=run_cell(nm,sim,cases,method,args.seed,calibration,identity);atomic(path,row)
        print('DONE',path.stem,row['return_mean'],flush=True)
    atomic(root/'diagnostics'/f'L1_s{args.seed}.json',diagnostics(nm,sim,cases,args.seed,calibration,identity))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',required=True);p.add_argument('--checkpoint-root',required=True)
    p.add_argument('--calibrate',action='store_true');p.add_argument('--calibration');p.add_argument('--source-commit');
    p.add_argument('--seed',type=int);p.add_argument('--device',default='cuda:0');p.add_argument('--users',type=int,default=16);args=p.parse_args()
    try:calibrate(args) if args.calibrate else run(args)
    except Exception as error:
        atomic(Path(args.out_dir)/f'failure_s{args.seed}_{time.time_ns()}.json',dict(error=repr(error),traceback=traceback.format_exc(),
            runtime=runtime_context(dict(torch=torch.__version__,numpy=np.__version__)),source=source_identity(),
            requested_source_commit=args.source_commit,calibration_path=args.calibration));raise
if __name__=='__main__':main()
