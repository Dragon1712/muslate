import itertools
import unittest
import numpy as np
from muslate.config import EnvConfig, CatalogConfig
from muslate.env.simulator import SlateSim
from muslate.planning.latent_model import OracleModel, CountingModel
from ops.r2a_planner import AddressedLookahead, node_seed, exact_oracle


class AllSlates:
    def __init__(self, model):
        self.model = model
        self.rng = np.random.default_rng(0)
        self.actions = np.array(list(itertools.permutations(range(model.n_items), model.slate_size)))
    def propose(self, states):
        return self.actions[None].copy()


class R2Tests(unittest.TestCase):
    def setUp(self):
        self.sim = SlateSim(EnvConfig(catalog=CatalogConfig(n_coarse=2,n_fine=2,dim=4),
                                    slate_size=2,horizon=3,n_users=2,div_penalty=.4))
        self.oracle = OracleModel(self.sim)
        self.state = self.sim.initial_state(0)
        self.latent = self.oracle.encode(self.state)[None]

    def test_address_has_no_depth_or_beam_and_order_matters(self):
        self.assertEqual(node_seed(42,((1,2),)),node_seed(42,((1,2),)))
        self.assertNotEqual(node_seed(42,((1,2),)),node_seed(42,((2,1),)))
        self.assertNotEqual(node_seed(42,()),node_seed(43,()))

    def test_same_node_candidates_after_other_searches(self):
        from muslate.planning.proposal import ProposalPolicy
        p=AddressedLookahead(self.oracle,ProposalPolicy(self.oracle,n_candidates=7,mode='uniform'),depth=2)
        root=p.pool(self.latent,991,())
        child=p.pool(self.latent,991,((0,1),))
        p.plan(self.latent,remaining_steps=3,decision_key=991)
        np.testing.assert_array_equal(root,p.pool(self.latent,991,()))
        np.testing.assert_array_equal(child,p.pool(self.latent,991,((0,1),)))

    def test_cached_root_identical_depth_beam_horizon(self):
        roots=AllSlates(self.oracle).actions
        for depth,beam in itertools.product(range(4),(3,8,12)):
            c=CountingModel(self.oracle)
            p=AddressedLookahead(c,AllSlates(c),depth=depth,beam_width=beam)
            a=p.plan(self.latent,remaining_steps=1,decision_key=2,root_candidates=roots)
            r=self.oracle.reward(np.repeat(self.latent,len(roots),axis=0),roots)
            np.testing.assert_array_equal(a[0],roots[np.argmax(r)])
            self.assertEqual(c.last_cost.transition_calls,len(roots))
            self.assertEqual(p.last['effective_depth'],0)
        with self.assertRaises(ValueError):p.plan(self.latent,remaining_steps=0)

    def test_exhaustive_matches_fullbeam_and_direct_enumeration(self):
        value,action,sequence=exact_oracle(self.sim,self.state,3)
        actions=AllSlates(self.oracle).actions
        best=-np.inf
        for seq in itertools.product(actions,repeat=3):
            state=self.state.clone();total=0
            for t,a in enumerate(seq):
                reward,state=self.sim.step(state,a,noise=False);total+=self.sim.cfg.gamma**t*reward
            best=max(best,total)
        self.assertAlmostEqual(value,best,places=10)
        p=AddressedLookahead(self.oracle,AllSlates(self.oracle),depth=2,beam_width=12,gamma=self.sim.cfg.gamma)
        p.plan(self.latent,remaining_steps=3)
        self.assertAlmostEqual(p.last['predicted_value'],value,places=10)
        self.assertEqual(len(sequence),3)


if __name__=='__main__':unittest.main()
