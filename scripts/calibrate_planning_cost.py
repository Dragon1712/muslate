"""Calibrate disjoint model and proposal costs on a declared hardware/batch."""
from pathlib import Path
import argparse,json,sys,time
from dataclasses import asdict
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from muslate.config import EnvConfig,CatalogConfig
from muslate.env.simulator import SlateSim
from muslate.model.checkpoint import load_checkpoint
from muslate.model.neural_model import NeuralLatentModel
from muslate.planning.latent_model import CountingModel
from scripts.run_gate3 import build_variants
from scripts.run_logged_pilot import atomic_json,runtime_info


def sync(model):
    device=getattr(model,'device',None)
    if device is not None and str(device).startswith('cuda'):torch.cuda.synchronize(device)


def calibrate(model,sim,reference_batch=128,repeats=20):
    if reference_batch<1 or repeats<3:raise ValueError('positive batch and >=3 repeats required')
    batches=sorted(set((1,16,reference_batch,512)))
    rng=np.random.default_rng(20261009);bench=[]
    for n in batches:
        s=np.zeros((n,model.state_dim))
        a=np.stack([rng.choice(model.n_items,size=model.slate_size,replace=False) for _ in range(n)])
        for name in ('reward','step','item_scores'):
            fn=getattr(model,name);args=(s,) if name=='item_scores' else (s,a)
            for _ in range(3):fn(*args)
            times=[]
            for _ in range(repeats):
                sync(model);start=time.perf_counter();fn(*args);sync(model)
                times.append(1000*(time.perf_counter()-start))
            bench.append(dict(operation=name,batch=n,median_ms=float(np.median(times)),
                              q10_ms=float(np.quantile(times,.1)),q90_ms=float(np.quantile(times,.9))))
    reference={b['operation']:b['median_ms']/reference_batch for b in bench if b['batch']==reference_batch}
    proposal={}
    # Exclusive proposal timing excludes the synchronized inner NumPy/Torch model calls.
    for name in ('flat_uniform','flat_policy','sid_flat','sid_hier'):
        samples=[]
        for i in range(repeats):
            counter=CountingModel(model);planner=build_variants(counter,sim,512,0,20261009+i)[name]
            planner.plan(np.zeros((1,model.state_dim)),remaining_steps=1)
            cost=counter.last_cost
            samples.append(cost.proposal_time_ms/max(cost.proposal_candidates,1))
        proposal[name]=dict(per_candidate_ms=float(np.median(samples)),q10=float(np.quantile(samples,.1)),q90=float(np.quantile(samples,.9)))
    common=float(np.median([v['per_candidate_ms'] for v in proposal.values()]))
    weights={name:dict(reward_ms=reference['reward'],transition_ms=reference['step'],
                       score_ms=reference['item_scores'],proposal_ms=proposal[name]['per_candidate_ms'] if name in proposal else common)
             for name in ('reactive','flat_uniform','flat_policy','sid_flat','sid_hier')}
    return dict(reference_batch=reference_batch,repeats=repeats,model_benchmarks=bench,
                proposal_benchmarks=proposal,method_weights=weights,
                scope='linear calibrated operation-cost estimate; actual end-to-end latency reported separately; reference batch scaling is approximate')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--out',required=True)
    ap.add_argument('--repeats',type=int,default=20);ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args();torch.set_num_threads(args.threads);device=torch.device(args.device)
    model,meta=load_checkpoint(args.checkpoint,device)
    env=dict(meta['identity']['config']['environment']);env['catalog']=CatalogConfig(**env['catalog'])
    result=calibrate(NeuralLatentModel([model],device),SlateSim(EnvConfig(**env)),repeats=args.repeats)
    result['runtime']=runtime_info(device);result['checkpoint']=str(Path(args.checkpoint).resolve())
    result['environment']=env|{'catalog':asdict(env['catalog'])}
    atomic_json(Path(args.out),result);print(json.dumps(result['method_weights']))


if __name__=='__main__':main()
