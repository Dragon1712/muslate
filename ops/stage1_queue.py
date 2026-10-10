"""Run only this isolated experiment's jobs; never manage other GPU processes."""
import argparse,json,os,subprocess,sys,threading,time,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_logged_pilot import build_pilot_arms,atomic_json
from scripts.reevaluate_stage1 import stage_source,require_selection,digest

def claim_queue(root,contract):
    path=Path(root)/f'{contract["phase"]}_queue_protocol.json'
    if path.is_file():
        if json.loads(path.read_text(encoding='utf-8'))!=contract:raise ValueError('queue protocol/source conflict')
    else:atomic_json(path,contract)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['validation','test'],required=True)
    p.add_argument('--out-dir',required=True);p.add_argument('--checkpoint-root',required=True)
    p.add_argument('--calibration',required=True);p.add_argument('--selection')
    p.add_argument('--gpus',type=int,nargs='+',default=list(range(6)))
    a=p.parse_args();root=Path(a.out_dir);root.mkdir(parents=True,exist_ok=True)
    arms=build_pilot_arms((1,2,3) if a.phase=='validation' else (1,2,3,4,5))
    if a.phase=='test':
        selected=require_selection(a.selection,stage_source())['selection']
        arms=[arm for arm in arms if arm['w_score']==selected[arm['condition']]]
    if not a.gpus or len(set(a.gpus))!=len(a.gpus):p.error('unique GPU IDs required')
    pending=list(arms);lock=threading.Lock();failures=[]
    contract=dict(phase=a.phase,arms=arms,source=stage_source(),checkpoint_root=a.checkpoint_root,
                  calibration_sha256=digest(a.calibration),selection_sha256=digest(a.selection) if a.phase=='test' else None)
    claim_queue(root,contract)
    attempt=root/'queue_attempts'/f'{a.phase}-{uuid.uuid4().hex}';attempt.mkdir(parents=True)
    atomic_json(attempt/'runtime.json',dict(gpus=a.gpus,pid=os.getpid(),
            started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    def worker(gpu):
        while True:
            with lock:
                if not pending:return
                arm=pending.pop(0)
            log=attempt/f'{arm["id"]}.log'
            cmd=[sys.executable,'scripts/reevaluate_stage1.py','--phase',a.phase,'--out-dir',str(root),
                 '--checkpoint-root',a.checkpoint_root,'--calibration',a.calibration,'--arm',arm['id'],
                 '--seeds',str(arm['seed']),'--device',f'cuda:{gpu}','--threads','4']
            if a.phase=='test':cmd+=['--selection',a.selection,'--train-missing']
            print(f'START {arm["id"]} gpu={gpu}',flush=True)
            with log.open('w',encoding='utf-8') as stream:
                env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env)
            print(f'DONE {arm["id"]} gpu={gpu} code={result.returncode}',flush=True)
            if result.returncode:
                with lock:failures.append(dict(arm=arm['id'],code=result.returncode,log=str(log)))
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        list(pool.map(worker,a.gpus))
    complete=dict(phase=a.phase,failures=failures,n_jobs=len(arms),attempt=str(attempt),
        finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    atomic_json(attempt/'complete.json',complete)
    marker=root/f'{a.phase}_queue_complete.json'
    if not failures and not marker.exists():atomic_json(marker,complete)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
