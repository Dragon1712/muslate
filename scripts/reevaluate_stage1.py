"""Stage1: reuse audited checkpoints, select on fresh validation, then new test."""
from pathlib import Path
from dataclasses import asdict
import argparse,copy,hashlib,json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from muslate.config import EnvConfig,CatalogConfig,WorldModelConfig,LogConfig
from muslate.env.simulator import SlateSim
from muslate.env.logger import generate_log,generate_observed_log
from muslate.model.train import train_world_model,train_logged_world_model
from muslate.model.checkpoint import load_checkpoint,save_checkpoint
from muslate.model.neural_model import NeuralLatentModel
from muslate.planning.latent_model import CountingModel
from muslate.planning.cost import CostWeights
from muslate.eval.rollout import build_warm_start_cases,evaluate_from_prefixes,make_learned_latent_fn
from muslate.eval.calibration import spearman
from muslate.eval.integrity import validate_protocols
from scripts.run_gate3 import build_variants
from scripts.run_logged_pilot import (atomic_json,run_lock,build_pilot_arms,validate_checkpoint,
                                      runtime_info,logged_diagnostics,log_digest)
from scripts.analyze_logged_pilot import validate_results,select_weights,normalized_contract,SELECTION_RULE


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_source():
    files=list((ROOT/'muslate').rglob('*.py'))
    files += [ROOT/'scripts'/n for n in ('reevaluate_stage1.py','calibrate_planning_cost.py',
               'analyze_logged_pilot.py','run_logged_pilot.py','run_gate3.py')]
    files += [ROOT/'ops'/'stage1_queue.py',ROOT/'docs'/'research'/'2026-10-09-stage1-protocol.md']
    return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(files)}


def fresh_seed(seed,split):
    if split not in ('validation','test') or seed not in range(1,6):raise ValueError('stage1 seed must be 1–5')
    return (10000000 if split=='validation' else 20000000)+2000*seed


def validate_origin(model,metadata,origin):
    validate_checkpoint(model,metadata,origin)
    c=origin['config'];arm=c['arm']
    if (model.cfg.oracle_state or model.cfg.seed!=arm['seed'] or model.cfg.w_score!=arm['w_score']
            or model.cfg.score_supervision!={'L0':'none','L1':'exposed','U1':'oracle_full'}[arm['condition']]):
        raise ValueError('invalid original information condition')


def require_selection(path,source):
    if not Path(path).is_file():raise ValueError('test requires frozen validation selection')
    s=json.loads(Path(path).read_text(encoding='utf-8'))
    validate_frozen_selection(s,source)
    return s


def validate_frozen_selection(s,source,validation_root=None):
    if s.get('source')!=source or s.get('selection_rule')!=SELECTION_RULE:
        raise ValueError('selection source/rule conflict')
    if set(s.get('selection',{}))!={'L0','L1','U1'} or s['selection']['L0']!=0:
        raise ValueError('invalid frozen condition matrix')
    if any(s['selection'][c] not in (1,10,100) for c in ('L1','U1')):
        raise ValueError('invalid frozen weights')
    entries=s.get('validation_files',[])
    if len(entries)!=21 or len({entry['path'] for entry in entries})!=21:
        raise ValueError('selection must pin21 unique validation files')
    records=[];arms=[]
    for entry in entries:
        original=Path(entry['path'])
        path=original if validation_root is None else Path(validation_root)/original.parent.name/original.name
        if not path.is_file() or digest(path)!=entry['sha256']:
            raise ValueError('frozen validation artifact changed')
        payload=json.loads(path.read_text(encoding='utf-8'));settings=payload['identity']['config']
        arm=settings['arm'];arms.append(arm)
        if (original.name!='validation.json' or original.parent.name!=arm['id']
                or settings.get('protocol_version')!=2):raise ValueError('invalid pinned validation identity')
        records.append(dict(identity=payload['identity'],protocols={'validation':payload['protocols']}))
    expected=build_pilot_arms((1,2,3))
    if sorted(arms,key=lambda a:a['id'])!=sorted(expected,key=lambda a:a['id']):
        raise ValueError('incomplete pinned validation arm matrix')
    actual_source,contract=validate_results(records,('validation',))
    if (actual_source!=source or s.get('contract')!=contract or s.get('seeds')!=[1,2,3]
            or select_weights(records)!=s['selection']):
        raise ValueError('frozen selection conflicts with raw validation source/contract/weights')
    return records


def select_stage1(out_dir,path):
    records=[];files=[]
    for arm in build_pilot_arms((1,2,3)):
        p=Path(out_dir)/arm['id']/'validation.json'
        if not p.is_file():raise ValueError(f'validation matrix incomplete: {arm["id"]}')
        s=json.loads(p.read_text(encoding='utf-8'))
        records.append(dict(identity=s['identity'],protocols={'validation':s['protocols']}))
        files.append(dict(path=str(p.resolve()),sha256=digest(p)))
    source,contract=validate_results(records,('validation',))
    if source!=stage_source() or contract.get('protocol_version')!=2:
        raise ValueError('selection requires current stage1 source/protocol')
    selection=select_weights(records)
    summary=dict(selection=selection,source=source,contract=contract,seeds=[1,2,3],
                 selection_rule=SELECTION_RULE,validation_files=files,
                 note='selected only after all21 fresh validation evaluations; no new test computed for selection')
    validate_frozen_selection(summary,source)
    atomic_json(Path(path),summary);print(json.dumps(selection));return summary


def get_checkpoint(arm,args,device):
    old=Path(args.checkpoint_root)/arm['id']/'model.pt'
    own=Path(args.out_dir)/'new_training'/arm['id']/'model.pt'
    path=old if old.is_file() else own
    if not path.is_file():
        if not args.train_missing:raise ValueError(f'missing selected checkpoint: {arm["id"]}')
        template=Path(args.checkpoint_root)/f'{arm["condition"]}_s1_w{arm["w_score"]}'/'model.pt'
        _,meta=load_checkpoint(template,torch.device('cpu'))
        origin=copy.deepcopy(meta['identity']);settings=origin['config']
        settings['arm']=arm;settings['model']['seed']=arm['seed'];settings['log']['seed']=10000+arm['seed']
        settings['planner_seed']=60000+arm['seed']
        origin['source']=stage_source()
        env=copy.deepcopy(settings['environment']);env['catalog']=CatalogConfig(**env['catalog'])
        sim=SlateSim(EnvConfig(**env));cfg=WorldModelConfig(**settings['model'])
        log_cfg=LogConfig(**settings['log'])
        log=generate_log(sim,log_cfg) if arm['condition']=='U1' else generate_observed_log(sim,log_cfg)
        start=time.perf_counter()
        model=(train_world_model(sim,log,cfg,device) if arm['condition']=='U1'
               else train_logged_world_model(log,cfg,device))
        if device.type=='cuda':torch.cuda.synchronize(device)
        meta=dict(identity=origin,train_seconds=time.perf_counter()-start,train_data_digest=log_digest(log),
                  runtime=runtime_info(device),note='new confirmation seed; same training architecture/data/optimizer budget as v1')
        save_checkpoint(model,own,meta);path=own
    model,meta=load_checkpoint(path,device);validate_origin(model,meta,meta['identity'])
    if meta['identity']['config']['arm']!=arm:raise ValueError('wrong checkpoint arm')
    return path,model,meta


def evaluate_protocol(nm,sim,settings,calibration,split,warmup):
    cases=build_warm_start_cases(sim,settings['users'],warmup,settings[f'{split}_prefix_seed'])
    counter=CountingModel(nm);planners=build_variants(counter,sim,settings['budget'],settings['depth'],settings['planner_seed'])
    rows={};latent_fn=make_learned_latent_fn(nm)
    for name,planner in planners.items():
        counter.weights=CostWeights(**calibration['method_weights'][name]);ranks=[];records=[]
        warm_planner=build_variants(counter,sim,settings['budget'],settings['depth'],settings['planner_seed'])[name]
        first=cases[0];warm_planner.plan(latent_fn(first.state,first.hist_slates,first.hist_rewards),remaining_steps=sim.cfg.horizon-first.state.t)
        if nm.device.type=='cuda':torch.cuda.synchronize(nm.device)
        counter.reset()
        def diagnostic(state,latent):
            ranks.append(float(spearman(nm.item_scores(latent)[0],sim.item_scores(state.h,state.sat))))
        def decision(user,state,slate,reward,used_planner):
            cost=asdict(counter.last_cost)
            records.append(dict(user_id=user,t=state.t,remaining_steps=sim.cfg.horizon-state.t,
                                effective_depth=0 if name=='reactive' else min(settings['depth'],sim.cfg.horizon-state.t-1),
                                slate=slate.tolist(),reward=reward,cost=cost,
                                estimated_compute_ms=counter.last_cost.estimate(counter.weights)))
        start=time.perf_counter()
        result=evaluate_from_prefixes(sim,planner,latent_fn,cases,observer=diagnostic,decision_observer=decision)
        costs={k:sum(record['cost'][k] for record in records) for k in asdict(counter.last_cost)}
        rows[name]=dict(return_mean=result.mean,return_se=result.se,tail_mean=result.tail_mean,tail_se=result.tail_se,
                        returns=result.returns.tolist(),tail_returns=result.tail_returns.tolist(),latency_ms=result.latency_ms,
                        n_decisions=result.n_decisions,slate_evals=counter.n_slate_evals,score_calls=counter.n_score_calls,
                        slate_evals_per_decision=counter.n_slate_evals/result.n_decisions,
                        closed_loop_spearman=float(np.mean(ranks)),closed_loop_spearman_per_user_step=ranks,
                        wall_seconds=time.perf_counter()-start,decisions=records,cost_totals=costs,
                        cost_weights=asdict(counter.weights),estimated_compute_ms=sum(r['estimated_compute_ms'] for r in records))
        print(f'  {split}/warm{warmup}/{name}: return={result.mean:.4f} decisions={result.n_decisions}',flush=True)
    return dict(warmup=warmup,user_ids=[c.user_id for c in cases],episode_seeds=[c.episode_seed for c in cases],
                prefix_returns=[c.prefix_return for c in cases],planner_seed=settings['planner_seed'],rows=rows)


def run_arm(arm,args,source,selection):
    device=torch.device(args.device);path,model,training=get_checkpoint(arm,args,device)
    settings=copy.deepcopy(training['identity']['config'])
    settings.update(users=args.users,holdout_episodes=args.holdout_episodes,warmup=args.warmup,
                    budget=args.budget,depth=args.depth,device=device.type,threads=args.threads,protocol_version=2,
                    planner_seed=60000+arm['seed'],
                    validation_seed=60000+arm['seed'],test_seed=70000+arm['seed'],
                    validation_prefix_seed=fresh_seed(arm['seed'],'validation'),test_prefix_seed=fresh_seed(arm['seed'],'test'),
                    calibration_sha256=digest(args.calibration),
                    feedback='synthetic continuous unary response, noise std .02',
                    budget_note='deduplicated pools; calibrated operation estimate and measured latency; no hard total cost matching in E1')
    if not 0<=settings['warmup']<settings['environment']['horizon']:raise ValueError('invalid warmup')
    identity=dict(config=settings,source=source,checkpoint_sha256=digest(path),training_identity=training['identity'])
    if selection is not None and normalized_contract(settings)!=selection['contract']:
        raise ValueError('test protocol conflicts with frozen validation contract')
    root=Path(args.out_dir)/arm['id'];root.mkdir(parents=True,exist_ok=True)
    manifest=root/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8'))!=identity:raise ValueError('evaluation manifest conflict')
    else:atomic_json(manifest,identity)
    calibration=json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    env=copy.deepcopy(settings['environment']);env['catalog']=CatalogConfig(**env['catalog']);sim=SlateSim(EnvConfig(**env))
    nm=NeuralLatentModel([model],device)
    with run_lock(root):
        saved={}
        for split in (('validation',) if args.phase=='validation' else ('validation','test')):
            stage=root/f'{split}.json'
            if stage.is_file():
                s=json.loads(stage.read_text(encoding='utf-8'))
                if s.get('identity')!=identity:raise ValueError('cached split identity conflict')
                validate_protocols({split:s.get('protocols',{})},settings,(split,))
                if split=='test' and s.get('selection_sha256')!=digest(args.selection):raise ValueError('test selection conflict')
            else:
                print(f'{arm["id"]}: {split} on {device}',flush=True)
                holdout=generate_log(sim,LogConfig(n_episodes=settings['holdout_episodes'],seed=settings[f'{split}_seed']))
                s=dict(identity=identity,diagnostics=logged_diagnostics(nm,sim,holdout),
                       runtime=runtime_info(device),
                       protocols={p:evaluate_protocol(nm,sim,settings,calibration,split,w)
                                  for p,w in (('full_episode',0),('common_prefix',settings['warmup']))})
                if split=='test':s['selection_sha256']=digest(args.selection)
                validate_protocols({split:s['protocols']},settings,(split,));atomic_json(stage,s)
            saved[split]=s
        if args.phase=='test':
            result=dict(identity=identity,status='complete',metadata=dict(train_seconds=training['train_seconds'],
                        checkpoint_path=str(path.resolve()),checkpoint_sha256=digest(path),training=training,
                        evaluation_runtime={k:s['runtime'] for k,s in saved.items()},calibration=calibration,selection_sha256=digest(args.selection)),
                        training_history=model.training_history,diagnostics={k:s['diagnostics'] for k,s in saved.items()},
                        protocols={k:s['protocols'] for k,s in saved.items()},
                        completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            validate_protocols(result['protocols'],settings)
            final=root/'result.json'
            if final.is_file():
                old=json.loads(final.read_text(encoding='utf-8'))
                for key in ('identity','metadata','training_history','diagnostics','protocols'):
                    if old.get(key)!=result[key]:raise ValueError('completed result dependency conflict')
            else:atomic_json(final,result)
        print(f'{arm["id"]}: {args.phase} COMPLETE',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=['validation','select','test'],required=True)
    ap.add_argument('--out-dir',required=True);ap.add_argument('--checkpoint-root');ap.add_argument('--calibration')
    ap.add_argument('--selection');ap.add_argument('--arm',default='all')
    ap.add_argument('--seeds',type=int,nargs='+',default=[1,2,3]);ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--users',type=int,default=16);ap.add_argument('--holdout-episodes',type=int,default=128)
    ap.add_argument('--warmup',type=int,default=5);ap.add_argument('--budget',type=int,default=512)
    ap.add_argument('--depth',type=int,default=2);ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--train-missing',action='store_true');args=ap.parse_args()
    if args.phase=='select':
        if not args.selection:ap.error('--selection required')
        select_stage1(args.out_dir,args.selection);return
    if not args.checkpoint_root or not args.calibration:ap.error('checkpoint-root/calibration required')
    if (len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0 or not 0<args.users<=64
            or args.holdout_episodes<1 or args.threads<1 or args.depth<0 or args.budget<40):ap.error('invalid experiment counts')
    if not set(args.seeds)<=set(range(1,4) if args.phase=='validation' else range(1,6)):
        ap.error('stage1 validation seeds1–3 / confirmation seeds1–5 required')
    source=stage_source();selection=None
    if args.phase=='test':
        if not args.selection:ap.error('test requires --selection')
        selection=require_selection(args.selection,source)
    arms=build_pilot_arms(args.seeds)
    if selection is not None:arms=[a for a in arms if a['w_score']==selection['selection'][a['condition']]]
    if args.arm!='all':arms=[a for a in arms if a['id']==args.arm]
    if not arms:ap.error('arm not in permitted phase matrix')
    torch.set_num_threads(args.threads)
    for arm in arms:run_arm(arm,args,source,selection)


if __name__=='__main__':main()
