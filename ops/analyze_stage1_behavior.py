"""Post-hoc descriptive decomposition of archived E1 executed actions."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from muslate.config import EnvConfig,CatalogConfig
from muslate.env.simulator import SlateSim
from muslate.eval.rollout import build_warm_start_cases
from scripts.analyze_logged_pilot import paired_interval

def main():
    base=ROOT/'results/e1_stage1_20261009';summary_path=base/'e1_summary_local.json'
    summary=json.loads(summary_path.read_text());tables={};intervals={};arrays={};n=0
    for condition,weight in summary['selection'].items():
        per_seed={};return_arrays={}
        for seed in summary['seeds']:
            path=base/'artifacts'/f'{condition}_s{seed}_w{weight}'/'test.json'
            payload=json.loads(path.read_text());cfg=payload['identity']['config']
            env=dict(cfg['environment']);env['catalog']=CatalogConfig(**env['catalog']);sim=SlateSim(EnvConfig(**env))
            protocol=payload['protocols']['common_prefix'];cases=build_warm_start_cases(sim,cfg['users'],protocol['warmup'],cfg['test_prefix_seed'])
            for planner,row in protocol['rows'].items():
                records={(d['user_id'],d['t']):d for d in row['decisions']};groups=[];sim_penalty=[];parts=[]
                for case in cases:
                    state=case.state.clone();total=np.zeros(3)
                    while state.t<sim.cfg.horizon:
                        d=records[(case.user_id,state.t)];slate=np.array(d['slate'],dtype=np.int64)
                        cats=sim.catalog.item_coarse[slate];z=sim.catalog.item_emb[slate]
                        affinity=float(np.einsum('k,kd,d->',sim.pos_w,z,state.h))
                        saturation=float(sim.lt*sim.cfg.sat_penalty*np.dot(sim.pos_w,state.sat[cats]))
                        diversity=float(sim.cfg.div_penalty*sim.diversity_term(slate)[0])
                        assert np.isclose(affinity-saturation-diversity,d['reward'],atol=1e-10,rtol=1e-10)
                        total+=sim.cfg.gamma**state.t*np.array([affinity,-saturation,-diversity])
                        groups.append(len(set(cats.tolist())));sim_penalty.append(diversity)
                        reward,state=sim.step(state,slate,noise=False);assert abs(reward-d['reward'])<1e-10;n+=1
                    parts.append(total)
                parts=np.asarray(parts)
                assert np.allclose(parts.sum(axis=1)+np.asarray(protocol['prefix_returns']),row['returns'],atol=1e-10,rtol=1e-10)
                per_seed.setdefault(planner,[]).append(dict(unique_coarse=float(np.mean(groups)),mean_diversity_penalty=float(np.mean(sim_penalty)),tail_affinity=float(parts[:,0].mean()),tail_saturation_contribution=float(parts[:,1].mean()),tail_diversity_contribution=float(parts[:,2].mean())))
                return_arrays.setdefault(planner,[]).append(row['returns'])
        tables[condition]={p:{k:float(np.mean([v[k] for v in values])) for k in values[0]} for p,values in per_seed.items()}
        arrays[condition]=return_arrays
        intervals[condition]={f'{left}-{right}':paired_interval(return_arrays[left],return_arrays[right]) for left,right in [('flat_policy','flat_uniform'),('sid_flat','flat_uniform')]}
    differences={}
    for label,left,right in [('L1_hier-L1_uniform',tables['L1']['sid_hier'],tables['L1']['flat_uniform']),('L1_hier-L0_hier',tables['L1']['sid_hier'],tables['L0']['sid_hier'])]:
        differences[label]={k:left[k]-right[k] for k in ('tail_affinity','tail_saturation_contribution','tail_diversity_contribution')}
    output=dict(scope='post-hoc descriptive common-prefix action/reward decomposition; true simulator values used only for evaluation; not causal module attribution',summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),n_replayed_decisions=n,per_condition=tables,exploratory_intervals=intervals,discounted_suffix_component_differences=differences)
    path=base/'detailed_behavior_analysis.json'
    with path.open('x',encoding='utf-8') as f:json.dump(output,f,indent=2)
    print(json.dumps(output,ensure_ascii=False))

if __name__=='__main__':main()
