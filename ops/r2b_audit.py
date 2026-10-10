"""Independent simulator replay and source/checkpoint/budget identity validation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from muslate.eval.rollout import build_warm_start_cases,make_learned_latent_fn
from ops.r2b_run import atomic,digest,source_identity,load,decision_key
from ops.r2b_search import METHODS,RANDOM_SEEDS,validate_ledger
from ops.r2b_identity import require_commit,validate_runtime,validate_command,gpu_for_seed

def verify_cell(cell,manifest,calibration_sha,nm,sim,cases,calibration):
    identity=cell['identity'];seed=cell['seed'];method=cell['method']
    if identity['source']!=manifest['source'] or identity['source_commit']!=manifest['source_commit'] or identity['calibration_sha256']!=calibration_sha:
        raise ValueError('cell freeze identity mismatch')
    if method not in METHODS or seed not in manifest['seeds']:raise ValueError('unexpected cell label')
    if cell['user_ids']!=[c.user_id for c in cases] or cell['episode_seeds']!=[c.episode_seed for c in cases]:raise ValueError('trajectory identity mismatch')
    latent_fn=make_learned_latent_fn(nm);count=0;errors=[]
    for index,case in enumerate(cases):
        if cell['prefix_actions'][index]!=[a.tolist() for a in case.hist_slates] or not np.allclose(cell['prefix_rewards'][index],case.hist_rewards,atol=1e-10,rtol=0):
            raise ValueError('shared prefix mismatch')
        state=case.state.clone();tail=0.;hs=[a.copy() for a in case.hist_slates];hr=list(case.hist_rewards)
        decisions=[d for d in cell['decisions'] if d['user_id']==case.user_id]
        if len(decisions)!=15:raise ValueError('wrong decision count')
        for d in decisions:
            if d['t']!=state.t or d['decision_key']!=decision_key(seed,case.user_id,state.t):raise ValueError('decision address mismatch')
            action=np.array(d['slate'],dtype=np.int64)
            if action.shape!=(sim.cfg.slate_size,) or len(set(action))!=len(action) or not np.all((action>=0)&(action<nm.n_items)):
                raise ValueError('illegal slate')
            p=d['planning']
            if p['base_cost_kind']!=('sid_ms' if 'sid' in method else 'uniform_ms'):raise ValueError('base coefficient method mismatch')
            cost=validate_ledger(p,calibration)
            if not np.isclose(cost,p['estimated_ms'],atol=1e-8,rtol=0) or cost>p['cap_ms']+1e-8 or not np.isclose(p['unused_ms'],p['cap_ms']-cost,atol=1e-8):
                raise ValueError('estimated budget mismatch')
            trace=p['incumbent_trace']
            if any(x>y for x,y in zip(trace,trace[1:])) or trace[-1]!=p['predicted_reward']:raise ValueError('nonmonotonic incumbent')
            latent=latent_fn(state,hs,hr);prediction=float(nm.reward(latent,action[None])[0])
            if not np.isclose(prediction,p['predicted_reward'],atol=2e-5,rtol=1e-5):raise ValueError('selected prediction mismatch')
            t=state.t;reward,state=sim.step(state,action,noise=False)
            if not np.isclose(reward,d['reward'],atol=1e-10,rtol=0):raise ValueError('reward replay mismatch')
            tail+=sim.cfg.gamma**t*reward;hs.append(action);hr.append(float(reward));count+=1
        if not np.isclose(tail,cell['tail_returns'][index],atol=1e-10,rtol=0) or not np.isclose(tail+case.prefix_return,cell['returns'][index],atol=1e-10,rtol=0):
            raise ValueError('discounted return mismatch')
    if not np.isclose(cell['return_mean'],np.mean(cell['returns']),atol=1e-10,rtol=0):raise ValueError('return mean mismatch')
    return dict(decisions=count,decision_errors=errors)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',required=True);p.add_argument('--checkpoint-root',required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    root=Path(args.out_dir);manifest=json.loads((root/'manifest.json').read_text())
    require_commit(manifest['source_commit'])
    if list(root.glob('failure*.json')):raise ValueError('failure artifacts present')
    validate_runtime(manifest['queue_runtime'],{'--out-dir':manifest['out_dir_argument'],'--checkpoint-root':manifest['checkpoint_root'],
         '--calibration':manifest['calibration_argument'],'--source-commit':manifest['source_commit']})
    if source_identity()!=manifest['source']:raise ValueError('source files changed since freeze')
    calibration_sha=digest(root/'calibration.json')
    calibration=json.loads((root/'calibration.json').read_text())
    if calibration_sha!=manifest['calibration_sha256']:raise ValueError('calibration content mismatch')
    files={p.stem:p for p in (root/'cells').glob('*.json')}
    if set(files)!=set(manifest['expected_cells']):raise ValueError('missing/unexpected cells')
    probes={p.stem:p for p in (root/'diagnostics').glob('*.json')}
    if set(probes)!=set(manifest['expected_diagnostics']):raise ValueError('missing/unexpected diagnostics')
    queue=json.loads((root/'queue_status.json').read_text())
    if queue['missing'] or queue['missing_diagnostics'] or any(j['exit_code'] for j in queue['jobs']):raise ValueError('queue incomplete/failed')
    if {j['seed'] for j in queue['jobs']}!=set(manifest['seeds']) or set(queue['complete'])!=set(manifest['expected_cells']):raise ValueError('queue identity mismatch')
    if len(queue['jobs'])!=len(manifest['seeds']):raise ValueError('duplicated queue jobs')
    launch_expected={}
    for job in queue['jobs']:
        seed=job['seed'];gpu=manifest['seed_to_gpu'][str(seed)]
        if gpu!=gpu_for_seed(seed):raise ValueError('manifest GPU assignment differs from frozen protocol')
        if job['gpu']!=gpu:raise ValueError('job assigned GPU mismatch')
        expected={'--out-dir':str(Path(manifest['out_dir_argument'])),'--checkpoint-root':manifest['checkpoint_root'],
          '--calibration':manifest['calibration_argument'],'--source-commit':manifest['source_commit'],
          '--seed':seed,'--device':f'cuda:{gpu}','--users':manifest['users']}
        validate_command(job['command'],expected);launch_expected[seed]=(expected,job['command'])
    reports={};hashes={};torch.set_num_threads(4)
    for seed in manifest['seeds']:
        nm,sim,current=load(args.checkpoint_root,seed,torch.device(args.device))
        cases=build_warm_start_cases(sim,manifest['users'],5,31000000+2000*seed)
        for method in METHODS:
            label=f'L1_s{seed}_{method}';cell=json.loads(files[label].read_text())
            if cell['seed']!=seed or cell['method']!=method:raise ValueError('cell label/content mismatch')
            runtime=cell['identity']['runtime'];validate_runtime(runtime,launch_expected[seed][0])
            if runtime['command']!=launch_expected[seed][1] or runtime['device']!=f"cuda:{manifest['seed_to_gpu'][str(seed)]}":raise ValueError('actual command/device mismatch')
            if not all(k in runtime['libraries'] for k in ('torch','numpy')):raise ValueError('missing library runtime metadata')
            if cell['identity']['checkpoint_sha256']!=current['checkpoint_sha256'] or cell['identity']['training_identity']!=current['training_identity']:
                raise ValueError('checkpoint identity mismatch')
            reports[label]=verify_cell(cell,manifest,calibration_sha,nm,sim,cases,calibration);hashes[str(files[label].relative_to(root))]=digest(files[label])
        probe=json.loads(probes[f'L1_s{seed}'].read_text())
        validate_runtime(probe['identity']['runtime'],launch_expected[seed][0])
        if probe['identity']['checkpoint_sha256']!=current['checkpoint_sha256']:raise ValueError('diagnostic checkpoint mismatch')
        if probe['seed']!=seed or len(probe['rows'])!=min(4,len(cases)):raise ValueError('diagnostic user count/seed mismatch')
        if probe['identity']['source']!=manifest['source'] or probe['identity']['source_commit']!=manifest['source_commit'] or probe['identity']['calibration_sha256']!=calibration_sha:raise ValueError('diagnostic freeze mismatch')
        for row,case in zip(probe['rows'],cases[:4]):
            if row['user_id']!=case.user_id:raise ValueError('diagnostic user mismatch')
            expected=('flat_coordinate_refine','true_sid_hier')+tuple(f'random_sid_hier_{s}' for s in RANDOM_SEEDS)
            if [t['method'] for t in row['trials']]!=list(expected):raise ValueError('diagnostic method/partition mismatch')
            anchors=[t['planning']['anchor'] for t in row['trials']]
            if any(a!=anchors[0] for a in anchors):raise ValueError('diagnostic anchors differ')
            for trial in row['trials']:
                validate_ledger(trial['planning'],calibration)
                if trial['planning']['base_cost_kind']!=('sid_ms' if 'sid' in trial['method'] else 'uniform_ms'):raise ValueError('diagnostic coefficient mismatch')
                r,_=sim.step(case.state.clone(),np.array(trial['slate']),noise=False)
                a,_=sim.step(case.state.clone(),np.array(anchors[0]),noise=False)
                if not np.isclose(r,trial['true_reward'],atol=1e-10) or not np.isclose(r-a,trial['true_improvement'],atol=1e-10):raise ValueError('diagnostic replay mismatch')
                latent=make_learned_latent_fn(nm)(case.state,case.hist_slates,case.hist_rewards)
                predictions=nm.reward(np.repeat(latent,2,axis=0),np.array([trial['slate'],anchors[0]]))
                if not np.isclose(predictions[0],trial['planning']['predicted_reward'],atol=2e-5,rtol=1e-5) or not np.isclose(predictions[1],trial['planning']['anchor_prediction'],atol=2e-5,rtol=1e-5):raise ValueError('diagnostic prediction mismatch')
                improvement=trial['planning']['predicted_reward']-trial['planning']['anchor_prediction']
                if not np.isclose(improvement,trial['predicted_improvement'],atol=1e-10) or not np.isclose(improvement-(r-a),trial['selection_bias'],atol=1e-10):raise ValueError('diagnostic improvement mismatch')
        hashes[str(probes[f'L1_s{seed}'].relative_to(root))]=digest(probes[f'L1_s{seed}'])
    hashes['manifest.json']=digest(root/'manifest.json');hashes['calibration.json']=calibration_sha;hashes['queue_status.json']=digest(root/'queue_status.json')
    atomic(root/'audit.json',dict(status='verified',cells=reports,input_sha256=hashes,source=source_identity(),
       n_cells=len(reports),n_decisions=sum(r['decisions'] for r in reports.values()),
       n_decision_errors=sum(len(r['decision_errors']) for r in reports.values())))
    print('VERIFIED',len(reports),'cells',sum(r['decisions'] for r in reports.values()),'decisions',flush=True)
if __name__=='__main__':main()
