"""Five-seed paired exploratory inference, with partition averaging within seed."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from ops.r2b_search import RANDOM_SEEDS
from ops.r2b_run import atomic,digest

CONTRASTS=(('flat_coordinate_refine','uniform'),('true_sid_hier','true_sid_flat'),
           ('random_sid_hier','random_sid_flat'),('true_sid_hier','flat_coordinate_refine'),
           ('true_sid_hier','random_sid_hier'),('true_sid_flat','random_sid_flat'))
def effect(diffs):
    seed_means=diffs.mean(axis=1);mean=float(seed_means.mean());margin=2.7764451051977987*float(seed_means.std(ddof=1))/np.sqrt(5)
    rng=np.random.default_rng(20261010);boot=[]
    for _ in range(10000):
        selected=rng.integers(5,size=5)
        boot.append(float(np.mean([diffs[s,rng.integers(diffs.shape[1],size=diffs.shape[1])].mean() for s in selected])))
    return dict(mean=mean,student_t95=[mean-margin,mean+margin],hierarchical_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),seed_means=seed_means.tolist())
def analyze(root):
    manifest=json.loads((root/'manifest.json').read_text());audit=json.loads((root/'audit.json').read_text())
    if manifest['seeds']!=[1,2,3,4,5]:raise ValueError('analysis requires full five seeds')
    if (audit.get('status')!='verified' or audit.get('n_decision_errors')!=0 or
        audit.get('n_cells')!=50 or audit.get('n_decisions')!=12000 or manifest['users']!=16 or
        set(audit.get('cells',{}))!=set(manifest['expected_cells'])):
        raise ValueError('analysis requires verified complete 50-cell / 12000-decision scope')
    cells={p.stem:json.loads(p.read_text()) for p in (root/'cells').glob('*.json')}
    for path,sha in audit['input_sha256'].items():
        if digest(root/path)!=sha:raise ValueError('audited input modified')
    values={};metrics={}
    for method in ('uniform','flat_coordinate_refine','true_sid_flat','true_sid_hier','random_sid_flat','random_sid_hier'):
        matrix=[];decisions=[]
        for seed in range(1,6):
            names=[f'{method}_{s}' for s in RANDOM_SEEDS] if method.startswith('random') else [method]
            members=[cells[f'L1_s{seed}_{name}'] for name in names]
            matrix.append(np.mean([c['returns'] for c in members],axis=0))
            decisions.extend(d for c in members for d in c['decisions'])
        values[method]=np.array(matrix)
        keys=set(k for d in decisions for k in d['planning']['cost'])
        metrics[method]=dict(return_mean=float(values[method].mean()),seed_return_means=values[method].mean(axis=1).tolist(),
            costs_per_decision={k:float(np.mean([d['planning']['cost'].get(k,0) for d in decisions])) for k in sorted(keys)},
            estimated_ms=float(np.mean([d['planning']['estimated_ms'] for d in decisions])),
            unused_ms=float(np.mean([d['planning']['unused_ms'] for d in decisions])),
            latency_ms=dict(mean=float(np.mean([d['latency_ms'] for d in decisions])),
                   p50=float(np.percentile([d['latency_ms'] for d in decisions],50)),p95=float(np.percentile([d['latency_ms'] for d in decisions],95))),
            stop_reasons={reason:sum(d['planning']['stop_reason']==reason for d in decisions) for reason in {d['planning']['stop_reason'] for d in decisions}},
            overestimation=float(np.mean([d['overestimation'] for d in decisions])))
    comparisons={f'{a}-minus-{b}':effect(values[a]-values[b]) for a,b in CONTRASTS}
    diagnostic={}
    for p in (root/'diagnostics').glob('*.json'):
        for row in json.loads(p.read_text())['rows']:
            for t in row['trials']:diagnostic.setdefault(t['method'],[]).append(t)
    summary={m:{k:float(np.mean([r[k] for r in rows])) for k in ('predicted_improvement','true_improvement','selection_bias')} for m,rows in diagnostic.items()}
    atomic(root/'analysis.json',dict(metrics=metrics,contrasts=comparisons,diagnostics=summary,
         calibration_sha256=manifest['calibration_sha256'],source_commit=manifest['source_commit'],audit_sha256=digest(root/'audit.json'),
         inference='5 independent training seeds; random partitions averaged within seed; exploratory unadjusted intervals'))
    print(json.dumps(comparisons,indent=2))
def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',required=True);args=p.parse_args();analyze(Path(args.out_dir))
if __name__=='__main__':main()
