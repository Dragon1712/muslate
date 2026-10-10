"""Independently replay recorded actions/rewards/returns, report complete matrix and paired seed CI."""
import argparse
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from muslate.config import EnvConfig,CatalogConfig
from muslate.env.simulator import SlateSim
from muslate.eval.rollout import build_warm_start_cases
from ops.run_r2a import atomic,digest,sequence_truth


def audit(root):
    root=Path(root);contract=json.loads((root/'protocol.json').read_text());cells={};checked=0
    for path in sorted((root/'cells').glob('*.json')):
        row=json.loads(path.read_text());identity=row['identity'];env=identity['environment'].copy()
        env['catalog']=CatalogConfig(**env['catalog']);sim=SlateSim(EnvConfig(**env))
        if identity['source']!=contract['source'] or identity['protocol_sha256']!=contract['protocol_sha256']:
            raise ValueError('source/protocol mismatch')
        if digest(identity['checkpoint_path'])!=identity['checkpoint_sha256']:raise ValueError('checkpoint changed')
        cases=build_warm_start_cases(sim,len(row['user_ids']),5,identity['prefix_seed'])
        grouped=defaultdict(list)
        for record in row['decisions']:grouped[record['user_id']].append(record)
        returns=[];tails=[]
        if row['n_decisions']!=len(cases)*15:raise ValueError('decision count mismatch')
        for case in cases:
            if row['episode_seeds'][case.user_id]!=case.episode_seed:raise ValueError('episode identity')
            if not np.isclose(row['prefix_returns'][case.user_id],case.prefix_return,atol=1e-10,rtol=0):raise ValueError('prefix return')
            state=case.state.clone();tail=0.
            records=grouped[case.user_id]
            if [r['t'] for r in records]!=list(range(5,20)):raise ValueError('missing/duplicate steps')
            for record in records:
                if record['planning']['effective_depth']!=min(row['depth'],20-state.t-1):raise ValueError('horizon mismatch')
                if len(record['planning']['predicted_sequence'])!=record['planning']['effective_depth']+1:raise ValueError('sequence horizon')
                value,rewards=sequence_truth(sim,state,record['planning']['predicted_sequence'])
                if not np.isclose(value,record['selected_sequence_true_value'],atol=1e-10,rtol=0):raise ValueError('sequence truth mismatch')
                t=state.t;reward,state=sim.step(state,np.array(record['slate']),noise=False)
                if not np.isclose(reward,record['reward'],atol=1e-10,rtol=0):raise ValueError('reward replay mismatch')
                tail+=sim.cfg.gamma**t*reward;checked+=1
            returns.append(case.prefix_return+tail);tails.append(tail)
        if not np.allclose(returns,row['returns'],atol=1e-10,rtol=0) or not np.allclose(tails,row['tail_returns'],atol=1e-10,rtol=0):raise ValueError('return replay mismatch')
        if not np.isclose(np.mean(returns),row['return_mean'],atol=1e-10,rtol=0):raise ValueError('return mean mismatch')
        cells[path.stem]=row
    missing=sorted(set(contract['expected_cells'])-set(cells));extra=sorted(set(cells)-set(contract['expected_cells']))
    probes={p.stem:json.loads(p.read_text()) for p in (root/'probes').glob('*.json')}
    expected_probes=contract.get('expected_probes',[])
    missing_probes=sorted(set(expected_probes)-set(probes));extra_probes=sorted(set(probes)-set(expected_probes))
    probe_states=0
    for name,probe in probes.items():
        if probe['identity']['source']!=contract['source']:raise ValueError('probe source mismatch')
        if len(probe['rows'])!=contract['probe_users'] or [r['user_id'] for r in probe['rows']]!=list(range(contract['probe_users'])):
            raise ValueError('probe user matrix mismatch')
        for row in probe['rows']:
            actions=np.array(row['root_candidates'],dtype='<i8');sha=hashlib.sha256(actions.tobytes()).hexdigest()
            if [(s['depth'],s['beam']) for s in row['searches']]!=[(0,3),(1,3),(2,3),(3,3),(1,8),(1,16)]:raise ValueError('probe search matrix mismatch')
            ref=row['reference'];values=np.array(ref['values']);rewards=np.array(ref['immediate_rewards']);order=np.argsort(-rewards,kind='stable')
            best=int(np.argmax(values));rank=int(np.flatnonzero(order==best)[0])+1
            if ref['best_index']!=best or ref['best_immediate_rank']!=rank:raise ValueError('probe reference rank mismatch')
            for search in row['searches']:
                if search['planning']['root_pool_sha256']!=sha:raise ValueError('cached root replay mismatch')
                if search['depth']==1:
                    maximum=float(np.max(values[order[:search['beam']]]))
                    if not np.isclose(maximum,search['planning']['predicted_value'],atol=1e-6,rtol=0):raise ValueError('probe beam reference mismatch')
                    if search['reference_best_pruned']!=(rank>search['beam']):raise ValueError('probe pruning mismatch')
            probe_states+=1
    queue_path=root/'queue_status.json';queue=json.loads(queue_path.read_text()) if queue_path.exists() else None
    queue_failed=queue is None or any(j['exit_code'] for j in queue['jobs'])
    tiny_path=root/'tiny_oracle.json';tiny=json.loads(tiny_path.read_text()) if tiny_path.exists() else None
    tiny_complete=tiny is not None and len(tiny['rows'])==4
    if tiny_complete:
        for row in tiny['rows']:
            if not np.isclose(row['exact_value'],row['trials'][-1]['return_value'],rtol=0,atol=1e-10):raise ValueError('tiny oracle mismatch')
    effects=[]
    for condition in ('L0','L1'):
        for proposal in ('flat_uniform','sid_hier'):
            for depth in (1,2,3):
                seed_differences=[];paired_users=[]
                for seed in range(1,6):
                    base=cells.get(f'{condition}_s{seed}_{proposal}_d0');row=cells.get(f'{condition}_s{seed}_{proposal}_d{depth}')
                    if base is None or row is None:continue
                    if base['episode_seeds']!=row['episode_seeds'] or base['prefix_returns']!=row['prefix_returns']:raise ValueError('paired identity mismatch')
                    differences=np.array(row['returns'])-np.array(base['returns'])
                    paired_users.append(differences);seed_differences.append(float(np.mean(differences)))
                if len(seed_differences)==5:
                    mean=float(np.mean(seed_differences));margin=2.7764451051977987*float(np.std(seed_differences,ddof=1))/np.sqrt(5)
                    # Supplementary paired resampling: seed, then users within selected seed.
                    rng=np.random.default_rng(20261009);boot=[]
                    for _ in range(10000):
                        sampled=[]
                        for i in rng.integers(0,5,size=5):
                            users=paired_users[i];sampled.append(float(np.mean(users[rng.integers(0,len(users),size=len(users))])))
                        boot.append(float(np.mean(sampled)))
                    effects.append(dict(condition=condition,proposal=proposal,depth=depth,seed_differences=seed_differences,
                                        mean=mean,ci95=[mean-margin,mean+margin],method='paired seed means Student-t df4',
                                        supplementary_hierarchical_bootstrap_ci95=np.percentile(boot,[2.5,97.5]).tolist(),
                                        bootstrap_replicates=10000,bootstrap_seed=20261009))
    summary=dict(complete_cells=len(cells),expected_cells=len(contract['expected_cells']),missing=missing,extra=extra,
                 replay_decisions=checked,complete_probe_files=len(probes),expected_probe_files=len(expected_probes),
                 probe_states=probe_states,missing_probes=missing_probes,extra_probes=extra_probes,
                 queue_failed=queue_failed,tiny_complete=tiny_complete,paired_depth_effects=effects,
                 files={p.name:digest(p) for p in (root/'cells').glob('*.json')})
    atomic(root/'audit.json',summary)
    print(json.dumps(summary,indent=2))
    if missing or extra or missing_probes or extra_probes or queue_failed or not tiny_complete:raise SystemExit(1)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();audit(args.root)
