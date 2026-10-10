"""Run additive R2-A cells and fixed-state probes, preserving frozen E1."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import itertools
import json
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
from muslate.planning.latent_model import CountingModel,OracleModel
from muslate.planning.proposal import ProposalPolicy
from muslate.planning.abstractions import SIDProposal
from muslate.eval.rollout import build_warm_start_cases,make_learned_latent_fn
from ops.r2a_planner import AddressedLookahead,exact_oracle


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def source_identity():
    files=list((ROOT/'muslate').rglob('*.py'))
    files+=[ROOT/'ops'/n for n in ('r2a_planner.py','run_r2a.py','r2a_queue.py','audit_r2a.py')]
    files+=[ROOT/'tests'/'test_r2a.py',ROOT/'docs/research/2026-10-09-r2a-protocol.md']
    return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(files)}


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def proposal(counter,sim,name):
    if name=='flat_uniform':return ProposalPolicy(counter,n_candidates=128,mode='uniform')
    if name=='sid_hier':return SIDProposal(counter,sim.n_coarse,sim.catalog.n_fine,
                                        n_sampled=34,out_size=34,widen=4,refine_positions=5,fine_per_pos=3)
    raise ValueError(name)


def sequence_truth(sim,state,sequence):
    total=0.;rewards=[];s=state.clone()
    for t,a in enumerate(sequence):
        r,s=sim.step(s,np.array(a,dtype=np.int64),noise=False)
        total+=sim.cfg.gamma**t*r;rewards.append(float(r))
    return float(total),rewards


def key(seed,user,t):return 30000000+100000*seed+100*user+t


def run_cell(nm,sim,cases,condition,seed,name,depth,identity):
    counter=CountingModel(nm);p=AddressedLookahead(counter,proposal(counter,sim,name),depth,3,sim.cfg.gamma)
    latent_fn=make_learned_latent_fn(nm);decisions=[];returns=[];tails=[]
    start=time.perf_counter()
    for case in cases:
        state=case.state.clone();hs=[a.copy() for a in case.hist_slates];hr=list(case.hist_rewards);tail=0.
        while state.t<sim.cfg.horizon:
            sync(nm.device);begin=time.perf_counter();latent=latent_fn(state,hs,hr);sync(nm.device)
            encoding_ms=1000*(time.perf_counter()-begin)
            begin=time.perf_counter();action=p.plan(latent,remaining_steps=sim.cfg.horizon-state.t,
                                                 decision_key=key(seed,case.user_id,state.t))[0].copy()
            sync(nm.device);latency_ms=1000*(time.perf_counter()-begin)
            truth,sequence_rewards=sequence_truth(sim,state,p.last['predicted_sequence'])
            old=state;r,state=sim.step(state,action,noise=False)
            decisions.append(dict(user_id=case.user_id,t=old.t,slate=action.tolist(),reward=float(r),
                                  remaining_steps=sim.cfg.horizon-old.t,decision_key=key(seed,case.user_id,old.t),
                                  planning=p.last.copy(),cost=asdict(counter.last_cost),
                                  latency_ms=latency_ms,encoding_ms=encoding_ms,
                                  selected_sequence_true_value=truth,selected_sequence_true_rewards=sequence_rewards,
                                  selected_sequence_overestimation=p.last['predicted_value']-truth))
            tail+=sim.cfg.gamma**old.t*r;hs.append(action);hr.append(float(r))
        tails.append(float(tail));returns.append(float(case.prefix_return+tail))
    costs={k:sum(r['cost'][k] for r in decisions) for k in asdict(counter.last_cost)}
    lat=np.array([r['latency_ms'] for r in decisions]);encoding=np.array([r['encoding_ms'] for r in decisions])
    return dict(identity=identity,status='complete',condition=condition,seed=seed,proposal=name,depth=depth,
                user_ids=[c.user_id for c in cases],episode_seeds=[c.episode_seed for c in cases],
                prefix_returns=[c.prefix_return for c in cases],returns=returns,tail_returns=tails,
                return_mean=float(np.mean(returns)),tail_mean=float(np.mean(tails)),decisions=decisions,
                n_decisions=len(decisions),cost_totals=costs,
                latency_ms=dict(mean=float(lat.mean()),p50=float(np.percentile(lat,50)),p95=float(np.percentile(lat,95))),
                encoding_ms=dict(mean=float(encoding.mean()),p50=float(np.percentile(encoding,50)),p95=float(np.percentile(encoding,95))),
                wall_seconds=time.perf_counter()-start)


def fixed_probe(nm,sim,case,seed,name):
    counter=CountingModel(nm);p=AddressedLookahead(counter,proposal(counter,sim,name),gamma=sim.cfg.gamma)
    latent=make_learned_latent_fn(nm)(case.state,case.hist_slates,case.hist_rewards)
    decision_key=key(seed,case.user_id,case.state.t)
    sync(nm.device);start=time.perf_counter()
    with counter.decision():root=p.pool(latent,decision_key,())
    sync(nm.device);cache_cost=asdict(counter.last_cost);cache_ms=1000*(time.perf_counter()-start)
    searches=[]
    for depth,beam in [(d,3) for d in range(4)]+[(1,8),(1,16)]:
        p.depth=depth;p.beam_width=beam
        sync(nm.device);start=time.perf_counter()
        action=p.plan(latent,remaining_steps=sim.cfg.horizon-case.state.t,decision_key=decision_key,root_candidates=root)[0]
        sync(nm.device);latency=1000*(time.perf_counter()-start)
        truth,_=sequence_truth(sim,case.state,p.last['predicted_sequence'])
        searches.append(dict(depth=depth,beam=beam,slate=action.tolist(),planning=p.last.copy(),
                             cost=asdict(counter.last_cost),latency_ms=latency,selected_sequence_true_value=truth))
    # Exact maximum over the same root pool and each root's complete successor pool.
    sync(nm.device);start=time.perf_counter()
    with counter.decision():
        rewards,nxt=counter.step(np.repeat(latent,len(root),axis=0),root)
        values=[];successor_identities=[];sequences=[]
        for i,a in enumerate(root):
            children=p.pool(nxt[i:i+1],decision_key,(tuple(map(int,a)),))
            child_rewards,_=counter.step(np.repeat(nxt[i:i+1],len(children),axis=0),children)
            j=int(np.argmax(child_rewards));values.append(float(rewards[i]+sim.cfg.gamma*child_rewards[j]))
            sequences.append([a.tolist(),children[j].tolist()])
            successor_identities.append(dict(action=a.tolist(),pool_sha256=hashlib.sha256(children.astype('<i8').tobytes()).hexdigest(),size=len(children)))
    sync(nm.device);ref_ms=1000*(time.perf_counter()-start);ref_cost=asdict(counter.last_cost)
    best=int(np.argmax(values));order=np.argsort(-rewards,kind='stable');rank=int(np.flatnonzero(order==best)[0])+1
    for search in searches:
        if search['depth']==1:
            allowed=order[:search['beam']]
            maximum=max(values[i] for i in allowed)
            if not np.isclose(maximum,search['planning']['predicted_value'],atol=1e-6):
                raise ValueError('beam subset/reference value mismatch')
            search.update(reference_best_pruned=bool(rank>search['beam']),
                          candidate_scope_regret=max(values)-search['planning']['predicted_value'])
    truth,_=sequence_truth(sim,case.state,sequences[best])
    return dict(user_id=case.user_id,decision_key=decision_key,root_candidates=root.tolist(),
                cache_generation_cost=cache_cost,cache_generation_latency_ms=cache_ms,
                searches=searches,reference=dict(scope='full cached roots and full addressed successor pools; learned model values',
                values=values,immediate_rewards=rewards.tolist(),best_index=best,best_immediate_rank=rank,
                best_sequence=sequences[best],best_sequence_true_value=truth,successor_pools=successor_identities,
                cost=ref_cost,latency_ms=ref_ms))


def tiny_diagnostic():
    sim=SlateSim(EnvConfig(catalog=CatalogConfig(n_coarse=2,n_fine=2,dim=4),slate_size=2,
                          horizon=3,n_users=4,div_penalty=.4))
    from tests.test_r2a import AllSlates
    oracle=OracleModel(sim);rows=[]
    for user in range(4):
        s=sim.initial_state(user);value,action,seq=exact_oracle(sim,s,3);latent=oracle.encode(s)[None]
        trials=[]
        for depth,beam in [(0,3),(2,3),(2,8),(2,12)]:
            p=AddressedLookahead(oracle,AllSlates(oracle),depth,beam,sim.cfg.gamma)
            chosen=p.plan(latent,remaining_steps=3)[0]
            total=0.;state=s.clone();executed=[]
            while state.t<3:
                chosen=p.plan(oracle.encode(state)[None],remaining_steps=3-state.t)[0]
                t=state.t;r,state=sim.step(state,chosen,noise=False);total+=sim.cfg.gamma**t*r
                executed.append(chosen.tolist())
            trials.append(dict(depth=depth,beam=beam,return_value=float(total),actions=executed))
        if not np.isclose(trials[-1]['return_value'],value,atol=1e-10):raise ValueError('exhaustive/full beam mismatch')
        rows.append(dict(user_id=user,exact_value=value,exact_action=action.tolist(),exact_sequence=seq,trials=trials))
    return dict(scope='4item K2 H3 all12 ordered legal slates and12^3 sequences, true dynamics',rows=rows)


def run_seed(args):
    out=Path(args.out_dir);source=source_identity();protocol_sha=digest(ROOT/'docs/research/2026-10-09-r2a-protocol.md')
    freeze=json.loads((out/'protocol.json').read_text())
    if freeze['source']!=source or freeze['protocol_sha256']!=protocol_sha:raise ValueError('frozen source/protocol conflict')
    device=torch.device(args.device);torch.set_num_threads(4)
    for condition,weight in [('L0',0),('L1',100)]:
        cp=Path(args.checkpoint_root)/f'{condition}_s{args.seed}_w{weight}'/'model.pt'
        model,meta=load_checkpoint(cp,device)
        arm=meta['identity']['config']['arm']
        if model.cfg.oracle_state or model.cfg.seed!=args.seed or model.cfg.w_score!=weight:
            raise ValueError('checkpoint information/seed/weight conflict')
        supervision={'L0':'none','L1':'exposed'}[condition]
        expected_arm={'id':f'{condition}_s{args.seed}_w{weight}','condition':condition,'seed':args.seed,
                      'w_score':weight,'oracle_state':False,'score_supervision':supervision}
        if (model.cfg.score_supervision!=supervision
            or any(arm.get(k)!=v for k,v in expected_arm.items())):
            raise ValueError('checkpoint arm/supervision metadata conflict')
        settings=copy.deepcopy(meta['identity']['config']['environment'])
        settings['catalog']=CatalogConfig(**settings['catalog']);sim=SlateSim(EnvConfig(**settings))
        if sim.cfg.horizon!=20:raise ValueError('R2 requires original20step environment')
        nm=NeuralLatentModel([model],device)
        identity=dict(source=source,protocol_sha256=protocol_sha,checkpoint_sha256=digest(cp),
                      checkpoint_path=str(cp),training_identity=meta['identity'],environment=asdict(sim.cfg),
                      runtime=dict(torch=torch.__version__,device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None),
                      prefix_seed=30000000+2000*args.seed)
        cases=build_warm_start_cases(sim,args.users,5,identity['prefix_seed'])
        # Warm the model/GPU before measured decisions, using a disjoint key.
        p=AddressedLookahead(nm,ProposalPolicy(nm,n_candidates=8,mode='uniform'))
        p.plan(make_learned_latent_fn(nm)(cases[0].state,cases[0].hist_slates,cases[0].hist_rewards),remaining_steps=15,decision_key=999)
        sync(device)
        for name,depth in itertools.product(('flat_uniform','sid_hier'),range(4)):
            cell=f'{condition}_s{args.seed}_{name}_d{depth}';path=out/'cells'/f'{cell}.json'
            if path.exists():raise ValueError(f'output already exists:{cell}')
            print(f'START {cell}',flush=True)
            row=run_cell(nm,sim,cases,condition,args.seed,name,depth,identity);atomic(path,row)
            print(f'DONE {cell} return={row["return_mean"]:.5f} latency_ms={row["latency_ms"]["mean"]:.2f}',flush=True)
            atomic(out/f'progress_s{args.seed}.json',dict(last_completed=cell,completed_files=len(list((out/'cells').glob(f'*_s{args.seed}_*.json')))))
        if args.probes:
            for name in ('flat_uniform','sid_hier'):
                rows=[]
                for case in cases[:4]:
                    print(f'PROBE {condition} s{args.seed} {name} u{case.user_id}',flush=True)
                    rows.append(fixed_probe(nm,sim,case,args.seed,name))
                atomic(out/'probes'/f'{condition}_s{args.seed}_{name}.json',dict(identity=identity,rows=rows))
    atomic(out/f'seed{args.seed}_complete.json',dict(status='complete',seed=args.seed,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',required=True);p.add_argument('--checkpoint-root',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--users',type=int,default=16)
    p.add_argument('--probes',action='store_true');args=p.parse_args()
    try:run_seed(args)
    except Exception as error:
        atomic(Path(args.out_dir)/f'seed{args.seed}_failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
