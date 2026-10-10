import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from muslate.planning.search import Lookahead,RandomShooting,CEM,MCTS,ReactiveGreedy,_MCTSNode


class TrapModel:
    state_dim=2;n_items=2;slate_size=1
    def __init__(self):self.seen=[]
    def item_scores(self,s):return np.tile([1.,0.],(len(s),1))
    def step(self,s,a):
        self.seen.extend(s[:,0].tolist())
        r=np.where(s[:,0]==19,1-a[:,0],np.where(s[:,0]>19,100*a[:,0],0.))
        return r,np.stack([s[:,0]+1,a[:,0]],axis=1)
    def reward(self,s,a):return self.step(s,a)[0]


class Pool:
    def propose(self,s):return np.tile(np.array([[[0],[1]]]),(len(s),1,1))


def planners(model):
    p=Pool()
    return [ReactiveGreedy(model),Lookahead(model,p,depth=5,beam_width=2),
            RandomShooting(model,p,depth=5,n_samples=64,seed=7),
            CEM(model,p,depth=5,n_samples=64,n_iters=2,seed=7),
            MCTS(model,p,max_depth=5,n_sims=10)]


class FiniteHorizonTests(unittest.TestCase):
    def test_standalone_oracle_probe_passes_actual_remaining_steps(self):
        from scripts.probe_environment import evaluate
        from muslate.config import EnvConfig,CatalogConfig
        from muslate.env import SlateSim
        from muslate.planning.latent_model import OracleModel
        sim=SlateSim(EnvConfig(catalog=CatalogConfig(n_coarse=2,n_fine=2),slate_size=2,horizon=3))
        class Planner:
            remaining=[]
            def plan(self,states,remaining_steps=None):
                self.remaining.append(remaining_steps)
                return np.array([[0,1]])
        p=Planner();evaluate(sim,OracleModel(sim),p,2,sim.cfg.gamma)
        self.assertEqual(p.remaining,[3,2,1,3,2,1])

    def test_small_oracle_matches_exhaustive_three_step_optimum(self):
        from itertools import permutations
        from muslate.config import EnvConfig,CatalogConfig
        from muslate.env.simulator import SlateSim
        from muslate.planning.latent_model import OracleModel
        sim=SlateSim(EnvConfig(catalog=CatalogConfig(n_coarse=2,n_fine=2,dim=4),slate_size=2,
                              horizon=3,long_term_strength=3,div_penalty=.8))
        actions=np.array(list(permutations(range(4),2)))
        class AllPool:
            def propose(self,s):return np.repeat(actions[None],len(s),axis=0)
        def value(state,left):
            if left==0:return 0.
            return max(sim.step(state,a,noise=False)[0]+sim.cfg.gamma*value(sim.step(state,a,noise=False)[1],left-1) for a in actions)
        state=sim.initial_state(0);oracle=OracleModel(sim)
        picked=Lookahead(oracle,AllPool(),depth=5,beam_width=12,gamma=sim.cfg.gamma).plan(oracle.encode(state)[None],remaining_steps=3)[0]
        reward,nxt=sim.step(state,picked,noise=False)
        self.assertAlmostEqual(reward+sim.cfg.gamma*value(nxt,2),value(state,3),places=10)

    def test_every_planner_uses_only_final_step_reward(self):
        for planner in planners(TrapModel()):
            with self.subTest(planner=type(planner).__name__):
                a=planner.plan(np.array([[19.,0.]]),remaining_steps=1)
                self.assertEqual(a.tolist(),[[0]])
                self.assertTrue(all(t==19 for t in planner.model.seen))

    def test_two_remaining_steps_never_evaluate_step20(self):
        for planner in planners(TrapModel()):
            with self.subTest(planner=type(planner).__name__):
                planner.plan(np.array([[18.,0.]]),remaining_steps=2)
                self.assertTrue(all(t<20 for t in planner.model.seen))

    def test_repeated_mcts_leaf_keeps_bootstrap(self):
        class Model:
            state_dim=1;n_items=1;slate_size=1
            def step(self,s,a):return s[:,0]+1,s+1
        class OnePool:
            def propose(self,s):return np.zeros((len(s),1,1),dtype=int)
        planner=MCTS(Model(),OnePool(),max_depth=1,gamma=1.)
        root=_MCTSNode(np.array([0.]));planner._expand(root)
        planner._simulate(root);planner._simulate(root)
        self.assertEqual(root.N.tolist(),[2.]);self.assertEqual(root.W.tolist(),[6.])

    def test_no_actions_remaining_is_rejected(self):
        for planner in planners(TrapModel()):
            with self.assertRaises(ValueError):planner.plan(np.array([[20.,0.]]),remaining_steps=0)


if __name__=='__main__':unittest.main()
