"""Pilot identity and model reuse are part of experimental reproducibility."""
from pathlib import Path
import sys
import tempfile
import unittest
import json
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muslate.config import WorldModelConfig
from muslate.model.world_model import WorldModel
from muslate.model.neural_model import NeuralLatentModel


class PilotTests(unittest.TestCase):
    def test_matrix_contains_21_distinct_arms(self):
        from scripts.run_logged_pilot import build_pilot_arms
        arms = build_pilot_arms()
        self.assertEqual(len(arms), 21)
        self.assertEqual(len({a['id'] for a in arms}), 21)
        for seed in (1, 2, 3):
            subset = [a for a in arms if a['seed'] == seed]
            self.assertEqual([a['condition'] for a in subset].count('L0'), 1)
            for a in subset:
                self.assertFalse(a['oracle_state'])
                self.assertEqual(a['score_supervision'], {'L0':'none','L1':'exposed','U1':'oracle_full'}[a['condition']])
            self.assertEqual(sorted(a['w_score'] for a in subset if a['condition']=='L1'), [1,10,100])

    def test_checkpoint_round_trip_preserves_predictions(self):
        from muslate.model.checkpoint import save_checkpoint, load_checkpoint
        torch.manual_seed(4)
        m = WorldModel(WorldModelConfig(oracle_state=False), 12, 3)
        m.training_history = [{'loss': 1.2}]
        nm = NeuralLatentModel([m])
        h = np.random.default_rng(4).normal(size=(2,64))
        sl = np.asarray([[0,1,2],[3,4,5]])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'model.pt'
            save_checkpoint(m,path,{'data_seed':41})
            restored, meta = load_checkpoint(path,torch.device('cpu'))
            with self.assertRaises(FileExistsError): save_checkpoint(m,path,{})
        self.assertEqual(meta,{'data_seed':41})
        self.assertEqual(restored.training_history,m.training_history)
        other = NeuralLatentModel([restored])
        np.testing.assert_array_equal(nm.item_scores(h),other.item_scores(h))
        np.testing.assert_array_equal(nm.reward(h,sl),other.reward(h,sl))
        for a,b in zip(nm.step(h,sl),other.step(h,sl)): np.testing.assert_array_equal(a,b)

    def test_resume_rejects_changed_config_or_source_hash(self):
        from scripts.run_logged_pilot import claim_run
        identity={'config':{'epochs':1},'source':{'file':'hash'}}
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(claim_run(Path(td),identity),'new')
            self.assertEqual(claim_run(Path(td),identity),'resume')
            for bad in ({'config':{'epochs':2},'source':identity['source']},
                        {'config':identity['config'],'source':{'file':'changed'}}):
                with self.assertRaises(ValueError): claim_run(Path(td),bad)

    def test_complete_result_cannot_be_overwritten(self):
        from scripts.run_logged_pilot import claim_run
        identity={'config':{'epochs':1},'source':{'file':'hash'}}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); claim_run(root,identity)
            raw=json.dumps({'identity':identity,'status':'complete'}).encode()
            (root/'result.json').write_bytes(raw)
            # A marker alone is no longer accepted as a complete experiment.
            with self.assertRaises(ValueError):claim_run(root,identity)
            self.assertEqual((root/'result.json').read_bytes(),raw)
            with self.assertRaises(ValueError): claim_run(root,{'config':{},'source':{}})

    def test_stale_lock_recovers_and_live_lock_blocks(self):
        from scripts.run_logged_pilot import run_lock
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'running.lock').write_text(json.dumps({'pid':999999999,'host':'old-host'}))
            with run_lock(root):
                with self.assertRaises(RuntimeError):
                    with run_lock(root): pass
            # Lock ownership is released even if the preceding process left a file.
            with run_lock(root): pass

    def test_resume_rejects_checkpoint_with_wrong_actual_model(self):
        from scripts.run_logged_pilot import validate_checkpoint
        cfg=WorldModelConfig(oracle_state=False,score_supervision='exposed',epochs=1)
        from dataclasses import asdict
        from muslate.config import EnvConfig
        identity={'config':{'model':asdict(cfg),'environment':asdict(EnvConfig()),'arm':{'condition':'L1'}}}
        model=WorldModel(cfg,200,5)
        validate_checkpoint(model,{'identity':identity},identity)
        for bad in (WorldModel(WorldModelConfig(oracle_state=False,epochs=2),200,5),
                    WorldModel(cfg,201,5),WorldModel(cfg,200,4)):
            with self.assertRaises(ValueError): validate_checkpoint(bad,{'identity':identity},identity)

    def test_evaluation_random_streams_do_not_overlap_between_runs(self):
        from scripts.run_logged_pilot import evaluation_seed
        streams=[]
        for split in ('validation','test'):
            for seed in (1,2,3):
                base=evaluation_seed(seed,split)
                current={base+1000+user for user in range(64)}
                for previous in streams:self.assertTrue(current.isdisjoint(previous))
                streams.append(current)


if __name__=='__main__': unittest.main()
