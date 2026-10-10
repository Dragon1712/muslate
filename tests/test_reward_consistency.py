import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from muslate.config import EnvConfig
from muslate.env.simulator import SlateSim


class RewardConsistencyTests(unittest.TestCase):
    def test_scalar_batch_consistency_across_diversity_and_positions(self):
        for div in (0.,.8):
            for discount in (.5,.8,1.):
                sim=SlateSim(EnvConfig(div_penalty=div,pos_discount=discount,long_term_strength=3))
                s=sim.initial_state(0);s.sat=np.linspace(0,1,sim.n_coarse);a=np.arange(5)
                with self.subTest(div=div,discount=discount):
                    self.assertAlmostEqual(sim.reward(s.h,s.sat,a),float(sim.reward_batch(s.h,s.sat,a)[0]),places=12)

    def test_scalar_noise_is_added_to_same_core(self):
        sim=SlateSim(EnvConfig(div_penalty=.8));s=sim.initial_state(0);a=np.arange(5)
        expected=float(sim.reward_batch(s.h,s.sat,a)[0])+float(np.random.default_rng(9).normal(0,.02))
        self.assertAlmostEqual(sim.reward(s.h,s.sat,a,noise=.02,rng=np.random.default_rng(9)),expected,places=12)


if __name__=='__main__':unittest.main()
