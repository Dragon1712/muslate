"""Export E1 effect intervals as a standalone publication figure."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summary',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args()
    data=json.loads(Path(args.summary).read_text(encoding='utf-8'))
    n=len(data['seeds']);colors={'L0':'#777777','L1':'#0072B2','U1':'#D55E00'}
    fig,axes=plt.subplots(1,2,figsize=(11,5.2),sharey=True,layout='constrained')
    for ax,protocol,title in zip(axes,('full_episode','common_prefix'),('Full episode','Shared 5-step prefix')):
        labels=[]
        for index,(condition,baseline) in enumerate((c,b) for c in ('L0','L1','U1') for b in ('flat_uniform','flat_policy','sid_flat')):
            result=data['protocols'][protocol]['comparisons'][f'{condition}:sid_hier-{baseline}']
            lo,hi=result['ci95'];color=colors[condition]
            ax.plot([lo,hi],[index,index],color=color,lw=1.8)
            ax.scatter(result['mean'],index,color=color,s=30,zorder=3)
            labels.append(f'{condition}: hier − {baseline}')
        ax.axvline(0,color='black',ls='--',lw=.8)
        ax.set_title(title,fontsize=11);ax.set_xlabel('Paired discounted return difference')
        ax.grid(axis='x',alpha=.2);ax.set_yticks(range(len(labels)),labels,fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle(f'E1: selected models, {n} independent seeds; hierarchical paired 95% intervals',fontsize=12)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.pdf'):
        target=path.with_suffix(suffix)
        if target.exists():raise FileExistsError(target)
        fig.savefig(target,dpi=220)
    plt.close(fig)
    print(path.with_suffix('.png'))


if __name__=='__main__':main()
