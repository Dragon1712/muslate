"""Descriptive measured-cost tables/figures for the frozen E1 test matrix."""
from pathlib import Path
import argparse,json
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--summary',required=True)
    ap.add_argument('--out',required=True);a=ap.parse_args()
    root=Path(a.root);s=json.loads(Path(a.summary).read_text());results={}
    for p in root.glob('*/result.json'):
        r=json.loads(p.read_text());arm=r['identity']['config']['arm']
        if arm['seed'] in s['seeds'] and arm['w_score']==s['selection'][arm['condition']]:
            results[(arm['condition'],arm['seed'])]=r
    if set(results)!={(c,seed) for c in s['selection'] for seed in s['seeds']}:raise ValueError('incomplete selected matrix')
    tables={}
    for protocol in s['protocols']:
        table={}
        for c in s['selection']:
            table[c]={}
            for planner in s['protocols'][protocol]['rows'][c]:
                rows=[results[(c,seed)]['protocols']['test'][protocol]['rows'][planner] for seed in s['seeds']]
                table[c][planner]=dict(return_mean=float(np.mean([r['return_mean'] for r in rows])),
                    latency_ms=float(np.mean([r['latency_ms'] for r in rows])),
                    estimated_compute_ms=float(np.mean([r['estimated_compute_ms']/r['n_decisions'] for r in rows])),
                    **{key:float(np.mean([r['cost_totals'][key]/r['n_decisions'] for r in rows])) for key in rows[0]['cost_totals']})
        tables[protocol]=table
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix('.json').open('x') as f:json.dump(dict(scope='descriptive E1; no strict hard compute matching; latency excludes belief encoding',protocols=tables),f,indent=2)
    lines=['# E1 分项成本（每决策，五seed均值）','',
        '估计毫秒使用实测系数；实际延迟为给定belief后的planner.plan，同步GPU。未施加严格等硬预算，不能由此声称E4效率优势。','']
    for protocol,table in tables.items():
        lines += [f'## {protocol}','','| 条件 | Planner | Return | Reward-only | Joint transition | Score | 生成尝试 | 有效交付 | 估计ms | 实际ms |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for c,rows in table.items():
            for planner,r in rows.items():
                vals=[r[k] for k in ('return_mean','reward_only_calls','transition_calls','score_queries','proposal_candidates','emitted_candidates','estimated_compute_ms','latency_ms')]
                lines.append('| '+c+' | '+planner+' | '+' | '.join(f'{x:.3f}' for x in vals)+' |')
        lines.append('')
    with out.with_suffix('.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    markers=dict(reactive='o',flat_uniform='v',flat_policy='s',sid_flat='D',sid_hier='*');colors=dict(L0='#777777',L1='#0072B2',U1='#D55E00')
    for row,(protocol,table) in enumerate(tables.items()):
        for col,xkey in enumerate(('estimated_compute_ms','latency_ms')):
            ax=axes[row,col]
            for c,rows in table.items():
                for planner,r in rows.items():
                    ax.scatter(r[xkey],r['return_mean'],color=colors[c],marker=markers[planner],s=65,label=f'{c}/{planner}')
            ax.set_title(protocol.replace('_',' '));ax.set_xlabel('Calibrated operation estimate (ms)' if col==0 else 'Measured planning latency (ms)')
            ax.set_ylabel('Discounted return');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=8)
    fig.suptitle('E1 descriptive return–cost; 5 training seeds, no strict hard budget matching')
    for suffix in ('.png','.pdf'):
        if out.with_suffix(suffix).exists():raise FileExistsError(out.with_suffix(suffix))
        fig.savefig(out.with_suffix(suffix),dpi=220)
    plt.close(fig)

if __name__=='__main__':main()
