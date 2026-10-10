"""Selection must use validation and uncertainty must respect model repeats."""
from pathlib import Path
import sys
import unittest
import numpy as np
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from muslate.config import EnvConfig,WorldModelConfig,LogConfig


class AnalysisTests(unittest.TestCase):
    def fixture(self,seeds=(1,2,3)):
        from scripts.run_logged_pilot import build_pilot_arms
        results=[]
        for arm in build_pilot_arms(seeds):
            cfg=WorldModelConfig(oracle_state=False,score_supervision=arm['score_supervision'],
                                 w_score=arm['w_score'],seed=arm['seed'])
            value=float(arm['w_score'])
            rows={p:dict(return_mean=value,returns=[value,value],tail_returns=[value,value],tail_mean=value,latency_ms=1.,
                        n_decisions=40,slate_evals_per_decision=4.,score_calls=1,
                        closed_loop_spearman=.5) for p in ('reactive','flat_uniform','flat_policy','sid_flat','sid_hier')}
            protocol=dict(warmup=0,rows=rows,user_ids=[0,1],episode_seeds=[arm['seed']*100,arm['seed']*100+1],prefix_returns=[0.,0.])
            results.append(dict(identity=dict(source={'model.py':'fixed'},config=dict(arm=arm,model=asdict(cfg),
                                environment=asdict(EnvConfig()),log=asdict(LogConfig(seed=arm['seed'])),
                                holdout_episodes=2,users=2,warmup=0,budget=512,depth=2)),
                                protocols={s:{p:protocol for p in ('full_episode','common_prefix')} for s in ('validation','test')},
                                diagnostics={'test':{'spearman_mean':.5,'reward_mae':.1}},metadata={'train_seconds':1.}))
        return results
    def test_hierarchical_paired_constant_difference(self):
        from scripts.analyze_logged_pilot import paired_interval
        left=np.arange(12).reshape(3,4)+2.
        result=paired_interval(left,left-2,draws=1000)
        self.assertEqual(result['mean'],2.)
        self.assertEqual(result['ci95'],[2.,2.])
        self.assertEqual(result['n_seeds'],3)
        with self.assertRaises(ValueError):paired_interval(left,left[:2])

    def test_selection_ignores_test_scores(self):
        from scripts.analyze_logged_pilot import select_weights
        results=[]
        for condition in ('L1','U1'):
            for seed in (1,2,3):
                for weight,val in ((1,2.),(10,3.),(100,3.)):
                    row={'return_mean':val,'returns':[val,val]}
                    results.append({'identity':{'config':{'arm':{'condition':condition,'seed':seed,'w_score':weight}}},
                        'protocols':{'validation':{'common_prefix':{'rows':{'sid_hier':row}}},
                                     'test':{'common_prefix':{'rows':{'sid_hier':{'return_mean':999 if weight==1 else -999}}}}}})
        self.assertEqual(select_weights(results),{'L0':0,'L1':10,'U1':10})

    def test_incompatible_unselected_training_run_is_rejected(self):
        from scripts.analyze_logged_pilot import build_summary
        results=self.fixture()
        results[1]['identity']['config']['log']['n_episodes']=1
        with self.assertRaises(ValueError):build_summary(results)

    def test_duplicate_seeds_are_not_independent_replicates(self):
        from scripts.analyze_logged_pilot import build_summary
        with self.assertRaises(ValueError):build_summary(self.fixture(),seeds=(1,1,2,3))

    def test_frozen_weights_require_full_source_and_protocol_provenance(self):
        from scripts.analyze_logged_pilot import build_summary
        with self.assertRaises(ValueError):build_summary(self.fixture(),frozen_selection={'L0':0,'L1':100,'U1':100})

    def test_malformed_user_pairing_is_rejected(self):
        from scripts.analyze_logged_pilot import build_summary
        results=self.fixture()
        for result in results:
            for p in ('full_episode','common_prefix'):
                result['protocols']['test'][p]['user_ids']=[0,0]
        with self.assertRaises(ValueError):build_summary(results)

    def test_frozen_selection_accepts_new_seeds_but_rejects_protocol_changes(self):
        from scripts.analyze_logged_pilot import build_summary
        original=build_summary(self.fixture())
        extra=[r for r in self.fixture((4,5)) if r['identity']['config']['arm']['w_score']==original['selection'][r['identity']['config']['arm']['condition']]]
        combined=self.fixture()+extra
        confirmed=build_summary(combined,seeds=(1,2,3,4,5),frozen_selection=original)
        self.assertEqual(confirmed['protocols']['common_prefix']['rows']['L1']['sid_hier']['n_seeds'],5)
        for field in ('source','budget'):
            changed=self.fixture()
            for result in changed:
                if field=='source':result['identity']['source']={'model.py':'other'}
                else:result['identity']['config']['budget']=256
            with self.assertRaises(ValueError):build_summary(changed,frozen_selection=original)


if __name__=='__main__': unittest.main()
