"""Plot observed reward differences; no causal module attribution."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1];base=root/'results/e1_stage1_20261009'
data=json.loads((base/'detailed_behavior_analysis.json').read_text())['per_condition']
summary=json.loads((base/'e1_summary_local.json').read_text())['protocols']['common_prefix']['comparisons']
fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharey=True,layout='constrained')
keys=('tail_affinity','tail_saturation_contribution','tail_diversity_contribution')
comparisons=[('L1 hier − L0 hier',data['L1']['sid_hier'],data['L0']['sid_hier'],'L1-L0:sid_hier'),('L1 hier − L1 SID flat',data['L1']['sid_hier'],data['L1']['sid_flat'],'L1:sid_hier-sid_flat')]
for ax,(title,left,right,key) in zip(axes,comparisons):
    delta=[left[k]-right[k] for k in keys];total=sum(delta);assert np.isclose(total,summary[key]['mean'],atol=1e-10)
    values=delta+[total];colors=['#0072B2' if v>=0 else '#D55E00' for v in delta]+['#222222']
    bars=ax.barh(range(4),values,color=colors,height=.55)
    for bar,v in zip(bars,values):
        ax.text(v+(.1 if v>=0 else -.1),bar.get_y()+bar.get_height()/2,f'{v:+.3f}',ha='left' if v>=0 else 'right',va='center',fontsize=9)
    ax.axvline(0,color='#444444',lw=.8);ax.set_xlim(-4.8,5.4);ax.set_title(title,fontsize=11)
    ax.set_yticks(range(4),['Preference match','Saturation term','Pairwise term','Net return']);ax.grid(axis='x',alpha=.15)
    ax.set_xlabel('Difference in discounted return');ax.spines[['top','right']].set_visible(False)
axes[0].invert_yaxis();fig.suptitle('Observed trajectory decomposition: shared 5-step prefix (post hoc)',fontsize=11)
out=base/'figures/stage1_reward_decomposition'
for suffix in ('.png','.pdf'):
    p=out.with_suffix(suffix)
    if p.exists():raise FileExistsError(p)
    fig.savefig(p,dpi=220)
plt.close(fig)
print(out.with_suffix('.png'))
