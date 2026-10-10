"""Export the predeclared primary effect and the auxiliary episode effect."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--summary',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    s=json.loads(Path(a.summary).read_text());fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
    for ax,p,title in zip(axes,('common_prefix','full_episode'),('Primary: shared 5-step prefix','Auxiliary: full episode')):
        r=s['protocols'][p]['comparisons']['L1-L0:sid_hier'];lo,hi=r['ci95']
        ax.scatter(r['seed_means'],range(1,6),color='#666666',s=25)
        ax.plot([lo,hi],[0,0],color='#0072B2',lw=2);ax.scatter(r['mean'],0,color='#0072B2',s=45)
        ax.set_yticks(range(6),['Paired mean']+[f'Training seed {n}' for n in s['seeds']]);ax.invert_yaxis()
        ax.axvline(0,color='black',ls='--',lw=.8);ax.grid(axis='x',alpha=.2)
        ax.set_xlim(min(0,lo)-.4,max(hi,max(r['seed_means']))+.4)
        ax.set_xlabel('L1 − L0 discounted return\n(SID hierarchical)',fontsize=9);ax.set_title(title,fontsize=10)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('E1: exposed-item supervision; 5 training seeds, paired 95% interval',fontsize=11)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.pdf'):
        if out.with_suffix(suffix).exists():raise FileExistsError(out.with_suffix(suffix))
        fig.savefig(out.with_suffix(suffix),dpi=220)
    plt.close(fig)

if __name__=='__main__':main()
