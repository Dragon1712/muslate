import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from muslate.config import EnvConfig, CatalogConfig
from muslate.env.simulator import SlateSim
from muslate.planning.latent_model import OracleModel
from muslate.planning.abstractions import SIDProposal
from muslate.planning.proposal import ProposalPolicy
from muslate.planning.sid_map import SIDMap


class SlateValidityTests(unittest.TestCase):
    def test_refinement_does_not_revaluate_duplicate_slates_at_same_state(self):
        sim=SlateSim(EnvConfig(long_term_strength=3.,div_penalty=.8))
        class TracedOracle(OracleModel):
            def __init__(self,sim):super().__init__(sim);self.rows=[]
            def reward(self,s,a):self.rows.extend(tuple(row) for row in a);return super().reward(s,a)
        model=TracedOracle(sim);prop=SIDProposal(model,10,20,n_sampled=136,out_size=136,widen=4,refine_positions=5,fine_per_pos=12)
        prop.propose(model.encode(sim.initial_state(0))[None])
        self.assertEqual(len(model.rows),len(set(model.rows)))

    def test_refinement_pool_is_legal_and_distinct(self):
        sim=SlateSim(EnvConfig(long_term_strength=3.,div_penalty=.8));model=OracleModel(sim)
        prop=SIDProposal(model,10,20,n_sampled=136,out_size=136,widen=4,
                         refine_positions=5,fine_per_pos=12,seed=0)
        rows=prop.propose(model.encode(sim.initial_state(0))[None])[0]
        self.assertTrue(all(len(set(a))==5 for a in rows))
        self.assertEqual(len({tuple(a) for a in rows}),len(rows))

    def test_capacity_shortfall_is_explicit(self):
        sim=SlateSim(EnvConfig(catalog=CatalogConfig(n_coarse=5,n_fine=1),slate_size=5))
        prop=SIDProposal(OracleModel(sim),sid_map=SIDMap.from_labels(np.arange(5)))
        with self.assertRaisesRegex(ValueError,'capacity'):
            prop._complete(np.arange(5.),np.zeros(5,dtype=int))

    def test_small_groups_are_sampled_without_capacity_violations(self):
        sim=SlateSim(EnvConfig(catalog=CatalogConfig(n_coarse=5,n_fine=1),slate_size=5))
        model=OracleModel(sim);prop=SIDProposal(model,5,1,n_sampled=100,out_size=100,refine_positions=5)
        rows=prop.propose(model.encode(sim.initial_state(0))[None])[0]
        self.assertTrue(all(len(set(a))==5 for a in rows))
        self.assertEqual(len({tuple(a) for a in rows}),len(rows))

    def test_full_catalog_flat_substitution_preserves_legal_action(self):
        sim=SlateSim(EnvConfig(catalog=CatalogConfig(n_coarse=5,n_fine=1),slate_size=5))
        model=OracleModel(sim);rows=ProposalPolicy(model,n_candidates=64,mode='item_perturb').propose(model.encode(sim.initial_state(0))[None])[0]
        self.assertTrue(all(len(set(a))==5 for a in rows))
        self.assertEqual(len(rows),1)

    def test_simulator_rejects_invalid_batch_and_scalar_actions(self):
        sim=SlateSim(EnvConfig());s=sim.initial_state(0)
        for bad in (np.array([0,0,2,3,4]),np.arange(4),np.array([0,1,2,3,200]),np.array([0.,1.,2.,3.,4.])):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):sim.step_batch(s.h,s.sat,bad[None])
                with self.assertRaises(ValueError):sim.step(s,bad,noise=False)


if __name__=='__main__':unittest.main()
