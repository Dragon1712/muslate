"""Supplementary verifier, independent of the frozen R2-A execution source.

Checks the binding of labels, checkpoints, shared case identities and regenerated
probe roots, then hashes every reporting input. Supports a portable checkpoint
root override. Frozen execution has a whole-run restart policy (no partial resume).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root,source_dir,checkpoint_root=None,device='cpu'):
    sys.path.insert(0,str(Path(source_dir).resolve()))
    import numpy as np
    import torch
    from muslate.config import EnvConfig,CatalogConfig
    from muslate.env.simulator import SlateSim
    from muslate.model.checkpoint import load_checkpoint
    from muslate.model.neural_model import NeuralLatentModel
    from muslate.planning.latent_model import CountingModel,OracleModel
    from muslate.eval.rollout import build_warm_start_cases,make_learned_latent_fn
    from ops.r2a_planner import AddressedLookahead,node_seed,exact_oracle
    from ops.run_r2a import proposal,key,atomic
    from tests.test_r2a import AllSlates
    torch.set_num_threads(4)
    root=Path(root);source_dir=Path(source_dir);contract=json.loads((root/'protocol.json').read_text())
    if contract['users']!=16 or contract['probe_users']!=4:raise ValueError('supplementary verifier requires full protocol')
    for name,sha in contract['source'].items():
        if digest(source_dir/name)!=sha:raise ValueError(f'frozen source changed:{name}')
    if digest(source_dir/'docs/research/2026-10-09-r2a-protocol.md')!=contract['protocol_sha256']:raise ValueError('protocol source changed')
    expected={f'{c}_s{s}_{p}_d{d}' for c in ('L0','L1') for s in range(1,6) for p in ('flat_uniform','sid_hier') for d in range(4)}
    expected_probes={f'{c}_s{s}_{p}' for c in ('L0','L1') for s in range(1,6) for p in ('flat_uniform','sid_hier')}
    if set(contract['expected_cells'])!=expected or set(contract['expected_probes'])!=expected_probes:raise ValueError('declared matrix')
    paths=list((root/'cells').glob('*.json'))
    if {p.stem for p in paths}!=expected:raise ValueError('actual cell matrix')
    checkpoints={};shared_cases={};shared_env=None;cells={}
    def bind(identity,condition,seed):
        nonlocal shared_env
        weight=0 if condition=='L0' else 100;supervision='none' if condition=='L0' else 'exposed'
        expected_arm=dict(id=f'{condition}_s{seed}_w{weight}',condition=condition,seed=seed,
                          w_score=weight,oracle_state=False,score_supervision=supervision)
        config=identity['training_identity']['config'];arm=config['arm']
        if any(arm.get(k)!=v for k,v in expected_arm.items()):raise ValueError('training arm identity')
        if identity['source']!=contract['source'] or identity['protocol_sha256']!=contract['protocol_sha256']:raise ValueError('artifact source/protocol')
        if identity['prefix_seed']!=30000000+2000*seed:raise ValueError('reserved trajectory identity')
        env=identity['environment']
        if shared_env is None:shared_env=env
        if env!=shared_env or env!=config['environment']:raise ValueError('shared environment/training mismatch')
        cp=(Path(checkpoint_root)/expected_arm['id']/'model.pt' if checkpoint_root else Path(identity['checkpoint_path']))
        token=(condition,seed)
        if token not in checkpoints:
            if digest(cp)!=identity['checkpoint_sha256']:raise ValueError('checkpoint SHA mismatch')
            model,metadata=load_checkpoint(cp,torch.device(device))
            if metadata['identity']!=identity['training_identity']:raise ValueError('checkpoint training metadata mismatch')
            if model.cfg.oracle_state or model.cfg.seed!=seed or model.cfg.w_score!=weight or model.cfg.score_supervision!=supervision:raise ValueError('loaded checkpoint condition')
            settings=env.copy();settings['catalog']=CatalogConfig(**settings['catalog']);sim=SlateSim(EnvConfig(**settings))
            cases=build_warm_start_cases(sim,16,5,identity['prefix_seed'])
            checkpoints[token]=(NeuralLatentModel([model],torch.device(device)),sim,cases,identity['checkpoint_sha256'],identity['training_identity'])
        nm,sim,cases,sha,training=checkpoints[token]
        if sha!=identity['checkpoint_sha256'] or training!=identity['training_identity']:raise ValueError('cross-artifact checkpoint conflict')
        return nm,sim,cases
    for path in sorted(paths):
        row=json.loads(path.read_text());condition=row['condition'];seed=row['seed'];name=row['proposal'];depth=row['depth']
        if path.stem!=f'{condition}_s{seed}_{name}_d{depth}' or path.stem not in expected:raise ValueError('cell label/stem mismatch')
        if row['status']!='complete':raise ValueError('cell incomplete')
        _,sim,cases=bind(row['identity'],condition,seed)
        ids=list(range(16));episodes=[c.episode_seed for c in cases];prefix=[c.prefix_return for c in cases]
        if row['user_ids']!=ids or row['episode_seeds']!=episodes or row['prefix_returns']!=prefix:raise ValueError('cell prefix case identity')
        if seed not in shared_cases:shared_cases[seed]=(ids,episodes,prefix)
        if (row['user_ids'],row['episode_seeds'],row['prefix_returns'])!=shared_cases[seed]:raise ValueError('paired case cross-condition mismatch')
        if row['n_decisions']!=240 or len(row['decisions'])!=240:raise ValueError('cell decision count')
        expected_steps={(u,t) for u in ids for t in range(5,20)}
        if {(d['user_id'],d['t']) for d in row['decisions']}!=expected_steps:raise ValueError('cell user/step matrix')
        for record in row['decisions']:
            u=record['user_id'];t=record['t'];k=key(seed,u,t);plan=record['planning']
            if record['decision_key']!=k or plan['root_seed']!=node_seed(k,()):raise ValueError('decision random identity')
            h=min(depth,19-t)
            if record['remaining_steps']!=20-t or plan['effective_depth']!=h or len(plan['predicted_sequence'])!=h+1:raise ValueError('record finite horizon')
            if record['slate']!=plan['predicted_sequence'][0]:raise ValueError('executed/selected action mismatch')
        cells[path.stem]=row
    probe_paths=list((root/'probes').glob('*.json'))
    if {p.stem for p in probe_paths}!=expected_probes:raise ValueError('actual probe matrix')
    replayed=0
    for path in sorted(probe_paths):
        condition,seed_token,name=path.stem.split('_',2);seed=int(seed_token[1:]);payload=json.loads(path.read_text())
        nm,sim,cases=bind(payload['identity'],condition,seed)
        if [r['user_id'] for r in payload['rows']]!=list(range(4)):raise ValueError('probe user labels')
        counter=CountingModel(nm);planner=AddressedLookahead(counter,proposal(counter,sim,name),gamma=sim.cfg.gamma)
        for row,case in zip(payload['rows'],cases):
            latent=make_learned_latent_fn(nm)(case.state,case.hist_slates,case.hist_rewards)
            k=key(seed,case.user_id,5)
            if row['decision_key']!=k:raise ValueError('probe decision key')
            regenerated=planner.pool(latent,k,());recorded=np.array(row['root_candidates'],dtype=np.int64)
            if not np.array_equal(regenerated,recorded):raise ValueError(f'probe proposal/condition/seed label mismatch:{path.stem}/u{case.user_id}')
            sha=hashlib.sha256(regenerated.astype('<i8').tobytes()).hexdigest()
            if [(s['depth'],s['beam']) for s in row['searches']]!=[(0,3),(1,3),(2,3),(3,3),(1,8),(1,16)]:raise ValueError('probe depth/beam matrix')
            for search in row['searches']:
                plan=search['planning']
                if plan['root_pool_sha256']!=sha or plan['root_seed']!=node_seed(k,()) or plan['root_pool_size']!=len(regenerated):raise ValueError('probe exact root identity')
                if search['slate']!=plan['predicted_sequence'][0] or plan['effective_depth']!=search['depth']:raise ValueError('probe selected action/horizon')
                if search['depth']==1:
                    values=np.array(row['reference']['values']);rewards=np.array(row['reference']['immediate_rewards'])
                    order=np.argsort(-rewards,kind='stable');maximum=float(np.max(values[order[:search['beam']]]))
                    expected_regret=float(np.max(values)-plan['predicted_value'])
                    if not np.isclose(maximum,plan['predicted_value'],rtol=0,atol=1e-6):raise ValueError('probe subset value')
                    if not np.isclose(expected_regret,search['candidate_scope_regret'],rtol=0,atol=1e-10):raise ValueError('probe regret field')
            replayed+=1
    queue=json.loads((root/'queue_status.json').read_text())
    if sorted(j['seed'] for j in queue['jobs'])!=list(range(1,6)) or any(j['exit_code'] for j in queue['jobs']):raise ValueError('queue job matrix/failure')
    if set(queue['complete'])!=expected or queue['missing']:raise ValueError('queue completion matrix')
    failures=list(root.glob('seed*_failure.json'))
    if failures:raise ValueError('failure markers present:'+','.join(p.name for p in failures))
    for seed in range(1,6):
        marker=json.loads((root/f'seed{seed}_complete.json').read_text())
        if marker['seed']!=seed or marker['status']!='complete':raise ValueError('seed completion marker')
    tiny=json.loads((root/'tiny_oracle.json').read_text())
    tiny_sim=SlateSim(EnvConfig(catalog=CatalogConfig(n_coarse=2,n_fine=2,dim=4),slate_size=2,horizon=3,n_users=4,div_penalty=.4))
    if [r['user_id'] for r in tiny['rows']]!=list(range(4)):raise ValueError('tiny user matrix')
    for row in tiny['rows']:
        value,action,seq=exact_oracle(tiny_sim,tiny_sim.initial_state(row['user_id']),3)
        if not np.isclose(value,row['exact_value'],atol=1e-10,rtol=0) or action.tolist()!=row['exact_action'] or seq!=row['exact_sequence']:raise ValueError('tiny exact replay')
        if [(t['depth'],t['beam']) for t in row['trials']]!=[(0,3),(2,3),(2,8),(2,12)]:raise ValueError('tiny trial matrix')
        oracle=OracleModel(tiny_sim)
        for trial in row['trials']:
            if len(trial['actions'])!=3:raise ValueError('tiny trial horizon')
            planner=AddressedLookahead(oracle,AllSlates(oracle),trial['depth'],trial['beam'],tiny_sim.cfg.gamma)
            state=tiny_sim.initial_state(row['user_id']);total=0.
            for t,recorded_action in enumerate(trial['actions']):
                selected=planner.plan(oracle.encode(state)[None],remaining_steps=3-t)[0]
                if selected.tolist()!=recorded_action:raise ValueError('tiny beam/myopic label/action replay')
                reward,state=tiny_sim.step(state,np.array(recorded_action,dtype=np.int64),noise=False)
                total+=tiny_sim.cfg.gamma**t*reward
            if not np.isclose(total,trial['return_value'],atol=1e-10,rtol=0):raise ValueError('tiny trial reward/return replay')
    # Every file the report consumes, plus queue/seed state, is now immutable by manifest.
    inputs=[root/'protocol.json',root/'audit.json',root/'queue_status.json',root/'tiny_oracle.json']+paths+probe_paths+list(root.glob('seed*_complete.json'))
    manifest={p.relative_to(root).as_posix():digest(p) for p in sorted(inputs)}
    result=dict(status='verified',source_manifest=contract['source'],verifier_sha256=digest(__file__),
                complete_cells=len(cells),complete_probe_files=len(probe_paths),regenerated_probe_roots=replayed,
                failure_markers=[],queue_jobs=queue['jobs'],portable_checkpoint_root=str(checkpoint_root) if checkpoint_root else None,
                runtime=dict(torch=torch.__version__,device=str(device),gpu=torch.cuda.get_device_name(torch.device(device)) if str(device).startswith('cuda') else None),
                restart_policy='whole new output directory; existing outputs are never resumed or overwritten',input_sha256=manifest)
    atomic(root/'verified_manifest.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('source_manifest','input_sha256')},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--source-dir',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--checkpoint-root');p.add_argument('--device',default='cpu');a=p.parse_args();verify(a.root,a.source_dir,a.checkpoint_root,a.device)
