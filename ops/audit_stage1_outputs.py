"""Independently replay environment rewards and verify frozen E1 provenance."""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.reevaluate_stage1 import stage_source,digest,require_selection,validate_origin,fresh_seed
from scripts.run_logged_pilot import atomic_json
from muslate.config import EnvConfig,CatalogConfig
from muslate.env import SlateSim
from muslate.eval.integrity import validate_protocols
from muslate.eval.rollout import build_warm_start_cases
from muslate.model.checkpoint import load_checkpoint

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--checkpoint-root',required=True)
    ap.add_argument('--selection',required=True);ap.add_argument('--source-manifest',required=True)
    a=ap.parse_args();torch.set_num_threads(4);root=Path(a.root);source=stage_source()
    assert source==json.loads(Path(a.source_manifest).read_text())['evaluation_source']
    frozen=require_selection(a.selection,source);sha=digest(a.selection)
    original=json.loads((root/'original_checkpoint_sha256.json').read_text())
    training_source=json.loads((root/'original_training_source_sha256.json').read_text())
    checkpoint_seen=set();n_decisions=0;n_splits=0;max_reward_error=0.;all_result_arms=set()
    for armroot in sorted(p for p in root.iterdir() if p.is_dir() and (p/'manifest.json').is_file()):
        identity=json.loads((armroot/'manifest.json').read_text());settings=identity['config'];arm=settings['arm']
        assert identity['source']==source and settings['calibration_sha256']==digest(root/'calibration_5090.json')
        path=Path(a.checkpoint_root)/arm['id']/'model.pt'
        if not path.is_file():path=root/'new_training'/arm['id']/'model.pt'
        assert digest(path)==identity['checkpoint_sha256']
        model,metadata=load_checkpoint(path,torch.device('cpu'));validate_origin(model,metadata,identity['training_identity'])
        assert model.cfg.n_ensemble==1 and not model.cfg.oracle_state
        assert len(model.training_history)==model.cfg.epochs==40
        assert all(torch.isfinite(p).all().item() for p in model.state_dict().values())
        if arm['id'] in original:
            assert digest(path)==original[arm['id']]
            assert metadata['identity']['source']==training_source
        else:assert metadata['identity']['source']==source
        checkpoint_seen.add(arm['id'])
        env=dict(settings['environment']);env['catalog']=CatalogConfig(**env['catalog']);sim=SlateSim(EnvConfig(**env))
        saved={}
        for split in ('validation','test'):
            f=armroot/f'{split}.json'
            if not f.is_file():
                assert split=='test' and (arm['seed'] not in (1,2,3,4,5) or arm['w_score']!=frozen['selection'][arm['condition']])
                continue
            d=json.loads(f.read_text());assert d['identity']==identity
            if split=='test':assert d['selection_sha256']==sha
            assert settings[f'{split}_prefix_seed']==fresh_seed(arm['seed'],split)
            validate_protocols({split:d['protocols']},settings,(split,));saved[split]=d;n_splits+=1
            for protocol,p in d['protocols'].items():
                cases=build_warm_start_cases(sim,settings['users'],p['warmup'],settings[f'{split}_prefix_seed'])
                assert p['user_ids']==[c.user_id for c in cases] and p['episode_seeds']==[c.episode_seed for c in cases]
                assert np.allclose(p['prefix_returns'],[c.prefix_return for c in cases],rtol=1e-12,atol=1e-12)
                for planner,row in p['rows'].items():
                    records={(d['user_id'],d['t']):d for d in row['decisions']};truth=[]
                    for case in cases:
                        state=case.state.clone();total=case.prefix_return
                        while state.t<sim.cfg.horizon:
                            record=records[(case.user_id,state.t)];t=state.t
                            reward,state=sim.step(state,np.asarray(record['slate'],dtype=np.int64),noise=False)
                            error=abs(reward-record['reward']);max_reward_error=max(max_reward_error,error)
                            assert error<1e-10
                            total+=sim.cfg.gamma**t*reward;n_decisions+=1
                        truth.append(total)
                    assert np.allclose(truth,row['returns'],rtol=1e-10,atol=1e-10)
        final=armroot/'result.json'
        if final.is_file():
            r=json.loads(final.read_text());assert r['status']=='complete' and r['identity']==identity
            assert r['metadata']['selection_sha256']==sha and r['metadata']['training']==metadata
            assert r['training_history']==model.training_history
            for split,d in saved.items():
                assert r['protocols'][split]==d['protocols'] and r['diagnostics'][split]==d['diagnostics']
                assert r['metadata']['evaluation_runtime'][split]==d['runtime']
            assert set(saved)=={'validation','test'}
            all_result_arms.add((arm['condition'],arm['seed'],arm['w_score']))
    assert all_result_arms=={(c,s,w) for c,w in frozen['selection'].items() for s in (1,2,3,4,5)}
    assert n_splits==42 and n_decisions==117600 and len(checkpoint_seen)==27
    report=dict(status='passed',n_checkpoints=len(checkpoint_seen),n_splits=n_splits,n_replayed_decisions=n_decisions,
        max_reward_error=max_reward_error,source=source,selection_sha256=sha,
        audit_script_sha256=digest(__file__),scope='case seeds/prefixes, legal actions/horizons/costs, actual environment reward replay, original checkpoint bytes/config/history/finite parameters, frozen validation/test binding')
    atomic_json(root/'final_integrity_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k!='source'}))

if __name__=='__main__':main()
