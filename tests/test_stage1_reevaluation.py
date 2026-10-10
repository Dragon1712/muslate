import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


class Stage1Tests(unittest.TestCase):
    def test_validation_only_reuses_checkpoint_without_creating_test(self):
        import argparse,json,torch
        from dataclasses import asdict
        from muslate.config import WorldModelConfig,EnvConfig,CatalogConfig,LogConfig
        from muslate.env.simulator import SlateSim
        from muslate.env.logger import generate_observed_log
        from muslate.model.train import train_logged_world_model
        from muslate.model.checkpoint import save_checkpoint
        from scripts.reevaluate_stage1 import run_arm,digest,stage_source
        from scripts.run_logged_pilot import build_pilot_arms
        torch.set_num_threads(1)
        arm=next(a for a in build_pilot_arms((1,)) if a['id']=='L1_s1_w1')
        env=EnvConfig(catalog=CatalogConfig(n_coarse=2,n_fine=2,dim=4),slate_size=2,horizon=3)
        cfg=WorldModelConfig(latent_dim=8,slate_emb_dim=4,hidden_dim=8,oracle_state=False,
                             score_supervision='exposed',n_ensemble=1,epochs=1,seed=1)
        model=train_logged_world_model(generate_observed_log(SlateSim(env),LogConfig(n_episodes=2)),cfg,torch.device('cpu'),False)
        settings=dict(arm=arm,model=asdict(cfg),environment=asdict(env),log=asdict(LogConfig(n_episodes=2)),planner_seed=60001)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);checkpoint=root/'old'/arm['id']/'model.pt'
            save_checkpoint(model,checkpoint,{'identity':{'config':settings,'source':{'fixture':'trained'}},'train_seconds':.1})
            before=digest(checkpoint)
            calibration=root/'calibration.json'
            calibration.write_text(json.dumps({'method_weights':{p:dict(reward_ms=2.,transition_ms=3.,score_ms=5.,proposal_ms=.5)
                for p in ('reactive','flat_uniform','flat_policy','sid_flat','sid_hier')}}))
            args=argparse.Namespace(checkpoint_root=str(root/'old'),out_dir=str(root/'new'),device='cpu',phase='validation',
                train_missing=False,users=1,holdout_episodes=2,warmup=1,budget=40,depth=2,threads=1,calibration=str(calibration))
            run_arm(arm,args,stage_source(),None)
            directory=root/'new'/arm['id'];self.assertFalse((directory/'test.json').exists());self.assertFalse((directory/'result.json').exists())
            saved=json.loads((directory/'validation.json').read_text());self.assertEqual(before,digest(checkpoint))
            for p in saved['protocols'].values():
                for row in p['rows'].values():
                    self.assertEqual(len(row['decisions']),row['n_decisions'])
                    self.assertTrue(all(len(set(r['slate']))==2 and r['effective_depth']<r['remaining_steps'] for r in row['decisions']))
            raw=(directory/'validation.json').read_bytes();run_arm(arm,args,stage_source(),None)
            self.assertEqual(raw,(directory/'validation.json').read_bytes())
            args.device='cpu:0'
            from unittest.mock import patch
            from scripts.reevaluate_stage1 import get_checkpoint
            with patch('scripts.reevaluate_stage1.get_checkpoint',side_effect=lambda arm,args,device:get_checkpoint(arm,args,torch.device('cpu'))):
                run_arm(arm,args,stage_source(),None)
            self.assertEqual(raw,(directory/'validation.json').read_bytes())

    def test_new_evaluation_seed_blocks_are_disjoint_from_old(self):
        from scripts.reevaluate_stage1 import fresh_seed
        from scripts.run_logged_pilot import evaluation_seed
        used=set()
        for seed in (1,2,3,4,5):
            for split in ('validation','test'):
                values={fresh_seed(seed,split)+1000+u for u in range(64)}
                self.assertTrue(used.isdisjoint(values));used|=values
                self.assertTrue(values.isdisjoint({evaluation_seed(seed,split)+1000+u for u in range(64)}))

    def test_test_phase_requires_frozen_compatible_selection(self):
        from scripts.reevaluate_stage1 import require_selection
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):require_selection(Path(d)/'missing.json',{})

    def test_changed_checkpoint_parameters_are_rejected(self):
        import torch
        from muslate.config import WorldModelConfig
        from muslate.model.world_model import WorldModel
        from scripts.reevaluate_stage1 import validate_origin
        from dataclasses import asdict
        cfg=WorldModelConfig(oracle_state=False,score_supervision='exposed',w_score=10,seed=2)
        m=WorldModel(cfg,200,5)
        origin={'config':{'model':asdict(cfg),'arm':{'condition':'L1','w_score':10,'seed':2},
                         'environment':{'catalog':{'n_coarse':10,'n_fine':20},'slate_size':5}}}
        validate_origin(m,{'identity':origin},origin)
        m.cfg.w_score=100
        with self.assertRaises(ValueError):validate_origin(m,{'identity':origin},origin)


if __name__=='__main__':unittest.main()
