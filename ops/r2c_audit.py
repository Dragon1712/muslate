"""Independent raw-metric, bank replay, checkpoint/label and completeness audit."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from muslate.model.checkpoint import load_checkpoint
from ops.r2c_run import digest,atomic,source_identity,simulator,validate_checkpoint
from ops.r2c_diagnostics import make_bank,bank_hash,summarize_case,aggregate


def require(condition,message):
    if not condition: raise ValueError(message)


def validate_model_identity(row,label,protocol,bank_entry,protocol_sha):
    require(label==f'{row["condition"]}_s{row["seed"]}','model label/stem mismatch')
    require(label in protocol['expected_models'],'unexpected model')
    require(row['status']=='complete' and row['source_commit']==protocol['source_commit'],'model status/source commit')
    identity=row['identity']
    require(identity['source']==protocol['source'],'model source hash')
    require(identity['protocol_file_sha256']==protocol_sha,'model protocol hash')
    require(identity['checkpoint']==protocol['checkpoints'][label],'model checkpoint binding')
    require(identity['training_identity']==protocol['checkpoints'][label]['training_identity'],'model training identity binding')
    arm=identity['training_identity']['config']['arm'];seed=row['seed'];condition=row['condition']
    weight=0 if condition=='L0' else 100;supervision='none' if condition=='L0' else 'exposed'
    expected=dict(id=f'{condition}_s{seed}_w{weight}',condition=condition,seed=seed,w_score=weight,
                  oracle_state=False,score_supervision=supervision)
    require(all(arm.get(k)==v for k,v in expected.items()),'model training arm identity')
    require(identity['bank']==bank_entry,'model bank binding')
    r=row['runtime']
    require(r['device']==f'cuda:{protocol["seed_to_gpu"][str(seed)]}' and bool(r['gpu']),'model runtime GPU')
    require(r['torch'] and r['numpy'] and r['python'] and r['executable'] and r['command'],'model runtime metadata')
    command=r['command']
    for flag,value in [('--seed',str(seed)),('--device',r['device']),('--source-commit',protocol['source_commit'])]:
        require(flag in command and command[command.index(flag)+1]==value,'model runtime command')


def finite_tree(value):
    if isinstance(value,float): require(np.isfinite(value),'nonfinite raw number')
    elif isinstance(value,dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value,list):
        for v in value: finite_tree(v)


def audit(root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());frozen=json.loads((root/'bank_manifest.json').read_text())
    require(protocol['format_version']==1,'protocol version')
    commit=protocol['source_commit']
    require(len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'source commit format')
    require(protocol['source']==source_identity(),'audit source hash mismatch')
    require(protocol['protocol_sha256']==digest(Path(__file__).resolve().parents[1]/'docs/research/2026-10-10-r2c-protocol.md'),'protocol document hash')
    seeds=[1] if protocol['smoke'] else list(range(1,6));users=1 if protocol['smoke'] else 16
    expected=sorted(f'{c}_s{s}' for c in ('L0','L1') for s in seeds)
    require(protocol['seeds']==seeds and protocol['n_users']==users,'protocol scope')
    require(protocol['expected_models']==expected and sorted(protocol['checkpoints'])==expected,'ten model checkpoint scope')
    require([protocol[k] for k in ('n_sequences','sequence_length','probe_count','horizons')]==[64,5,16,[1,3,5]],'bank design scope')
    require(frozen['protocol_sha256']==digest(root/'protocol.json'),'bank manifest protocol hash')
    require(sorted(frozen['banks'])==list(map(str,seeds)),'bank completeness')
    require(sorted(p.stem for p in (root/'models').glob('*.json'))==expected,'model completeness')
    queue=json.loads((root/'queue_status.json').read_text())
    require(queue['present']==expected and not queue['missing'],'queue completeness')
    require(sorted(j['seed'] for j in queue['jobs'])==seeds and all(j['exit_code']==0 for j in queue['jobs']),'queue jobs')
    require(not list(root.glob('failure_*.json')),'failure artifact present')
    inputs={'protocol.json':digest(root/'protocol.json'),'bank_manifest.json':digest(root/'bank_manifest.json'),
            'queue_status.json':digest(root/'queue_status.json')}
    sim=simulator(protocol['environment']);banks={}
    for seed in seeds:
        entry=frozen['banks'][str(seed)];path=root/entry['path'];bank=json.loads(path.read_text())
        require(entry['path']==f'banks/seed{seed}.json','bank path identity')
        require(digest(path)==entry['sha256'] and bank_hash(bank)==entry['content_sha256'],'bank hash')
        print(f'AUDIT replay bank seed={seed}',flush=True)
        regenerated=make_bank(sim,seed,users)
        require(bank_hash(regenerated)==bank_hash(bank),'bank/evaluator replay mismatch')
        finite_tree(bank);banks[seed]=bank;inputs[entry['path']]=digest(path)
        complete=root/f'seed{seed}_complete.json'
        require(json.loads(complete.read_text())==dict(status='complete',seed=seed),'seed completion marker')
        inputs[complete.name]=digest(complete)
        log=root/f'seed{seed}.log';require(log.is_file(),'seed log missing');inputs[log.name]=digest(log)
    ncases=0;undefined=0
    for label in expected:
        path=root/'models'/f'{label}.json';row=json.loads(path.read_text());seed=int(label.split('_s')[1])
        validate_model_identity(row,label,protocol,frozen['banks'][str(seed)],digest(root/'protocol.json'))
        cp_entry=protocol['checkpoints'][label];cp=Path(cp_entry['path'])
        require(digest(cp)==cp_entry['sha256'],'audit checkpoint hash')
        model,meta=load_checkpoint(cp,torch.device('cpu'));validate_checkpoint(model,meta,row['condition'],seed)
        require(meta['identity']==cp_entry['training_identity'],'audit checkpoint training identity')
        require(meta['identity']['config']['environment']==protocol['environment'],'audit checkpoint environment')
        finite_tree(row)
        require(len(row['cases'])==users and [c['user_id'] for c in row['cases']]==list(range(users)),'model user completeness')
        for case,bankcase in zip(row['cases'],banks[seed]['cases']):
            require(case['initial_state_sha256']==bankcase['initial_state_sha256'] and
                    case['history_sha256']==bankcase['history_sha256'],'model initial state/history binding')
            pred={key:np.asarray(value) for key,value in case['prediction'].items()}
            require(set(pred)=={'open_loop','teacher_forced','probe_open_loop','probe_teacher_forced'},'prediction mode scope')
            for key,value in pred.items():
                shape=(64,5,16) if key.startswith('probe_') else (64,5)
                require(value.shape==shape and np.isfinite(value).all(),'prediction shape/finiteness')
            require(np.array_equal(pred['open_loop'][:,0],pred['teacher_forced'][:,0]),'horizon1 mode equality')
            require(np.array_equal(pred['probe_open_loop'][:,0],pred['probe_teacher_forced'][:,0]),'horizon1 probe equality')
            metrics=summarize_case(bankcase,pred,sim.cfg.gamma)
            require(case['metrics']==metrics,'raw metric recomputation')
            undefined+=sum(metrics['horizons'][h][m]['rank_correlation'] is None
                           for h in ('1','3','5') for m in ('open_loop','teacher_forced'))
            ncases+=1
        require(row['aggregate']==aggregate(row['cases']),'aggregate metric recomputation')
        inputs[path.relative_to(root).as_posix()]=digest(path)
        print(f'AUDIT model={label} complete',flush=True)
    manifest=dict(status='verified',smoke=protocol['smoke'],source_commit=commit,expected_models=expected,
                  model_count=len(expected),case_count=ncases,undefined_rank_count=undefined,
                  input_sha256=inputs,auditor_sha256=digest(__file__),
                  checks=['checkpoint labels/config/hash','source/protocol/bank hash','complete scope',
                          'regenerated deterministic banks and simulator truths','raw metric and aggregate replay',
                          'same selected sequence in both modes','finiteness with explicit undefined ranks',
                          'horizon1 reward and probe equality','runtime GPU and actual command'],
                  verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    atomic(root/'verified_manifest.json',manifest);print(json.dumps(manifest,ensure_ascii=False),flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();audit(args.root)
