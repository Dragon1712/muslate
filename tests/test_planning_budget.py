import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from muslate.planning.latent_model import CountingModel
from muslate.planning.search import Lookahead
from muslate.planning.proposal import ProposalPolicy


class Model:
    n_items=10;slate_size=2;state_dim=1
    def __init__(self):self.steps=0;self.scores=0
    def reward(self,s,a):return a.sum(axis=1).astype(float)
    def step(self,s,a):self.steps+=len(a);return self.reward(s,a),s+1
    def item_scores(self,s):self.scores+=len(s);return np.repeat(np.arange(10.)[None],len(s),axis=0)


class PlanningBudgetTests(unittest.TestCase):
    def test_hard_budget_rejects_unbound_proposal_before_any_work(self):
        from muslate.planning.cost import CostWeights,PlanningBudget
        for raw in (True,False):
            inner=Model();c=CountingModel(inner)
            other=inner if raw else CountingModel(inner)
            planner=Lookahead(c,ProposalPolicy(other,n_candidates=4),depth=0)
            with self.assertRaisesRegex(ValueError,'proposal'):
                planner.plan(np.zeros((1,1)),remaining_steps=1,budget=PlanningBudget(0.,CostWeights(2.,3.,5.,.5)))
            self.assertEqual(inner.scores,0)

    def test_hard_budget_rejects_unknown_external_proposal_even_with_model_field(self):
        from muslate.planning.cost import CostWeights,PlanningBudget
        inner=Model();c=CountingModel(inner)
        class External:
            model=c
            def propose(self,states):
                inner.item_scores(states)
                return np.array([[[0,1]]])
        planner=Lookahead(c,External())
        with self.assertRaisesRegex(ValueError,'proposal'):
            planner.plan(np.zeros((1,1)),remaining_steps=1,budget=PlanningBudget(0.,CostWeights(2.,3.,5.,.5)))
        self.assertEqual(inner.scores,0)

    def test_cem_sampling_counts_unique_emitted_pool_and_time(self):
        from muslate.planning.search import CEM
        inner=Model();c=CountingModel(inner)
        planner=CEM(c,ProposalPolicy(c),depth=0,n_samples=32,n_iters=1)
        planner.plan(np.zeros((1,1)),remaining_steps=1)
        self.assertEqual(c.last_cost.proposal_candidates,32)
        self.assertGreater(c.last_cost.emitted_candidates,0)
        self.assertGreater(c.last_cost.proposal_time_ms,0)
        self.assertEqual(c.last_cost.emitted_candidates,c.last_cost.unique_slates)

    def test_calibration_uses_measured_positive_reference_costs(self):
        from scripts.calibrate_planning_cost import calibrate
        from muslate.config import EnvConfig,CatalogConfig
        from muslate.env.simulator import SlateSim
        sim=SlateSim(EnvConfig(catalog=CatalogConfig(n_coarse=5,n_fine=2),slate_size=2))
        result=calibrate(Model(),sim,reference_batch=16,repeats=3)
        for w in result['method_weights'].values():
            self.assertTrue(all(x>0 and np.isfinite(x) for x in w.values()))
            self.assertNotEqual(list(w.values()),[1.,1.,1.,1.])
        self.assertEqual(result['reference_batch'],16)

    def test_counts_are_disjoint_and_weighted_by_measured_coefficients(self):
        from muslate.planning.cost import CostWeights
        c=CountingModel(Model(),weights=CostWeights(2.,3.,5.,.5))
        c.reward(np.zeros((2,1)),np.array([[0,1],[1,2]]))
        c.step(np.zeros((3,1)),np.array([[0,1],[1,2],[2,3]]))
        c.item_scores(np.zeros((1,1)));c.reserve_proposals(4)
        self.assertEqual((c.cost.reward_only_calls,c.cost.transition_calls,c.cost.score_queries),(2,3,1))
        self.assertEqual(c.cost.unique_slates,3)
        self.assertAlmostEqual(c.estimated_compute_ms,20.)

    def test_budget_refuses_call_before_model_work(self):
        from muslate.planning.cost import CostWeights,PlanningBudget,BudgetExceeded
        inner=Model();c=CountingModel(inner)
        planner=Lookahead(c,ProposalPolicy(c,n_candidates=4,mode='uniform'),depth=2)
        budget=PlanningBudget(7.,CostWeights(2.,3.,5.,.5))
        with self.assertRaises(BudgetExceeded):planner.plan(np.zeros((1,1)),remaining_steps=3,budget=budget)
        self.assertLessEqual(budget.used,7.)
        self.assertEqual(inner.steps,0)
        self.assertEqual(c.last_cost.proposal_candidates,4)

    def test_proposal_reservation_cannot_escape_budget(self):
        from muslate.planning.cost import CostWeights,PlanningBudget,BudgetExceeded
        inner=Model();c=CountingModel(inner)
        planner=Lookahead(c,ProposalPolicy(c,n_candidates=4),depth=0)
        budget=PlanningBudget(1.,CostWeights(2.,3.,5.,.5))
        with self.assertRaises(BudgetExceeded):planner.plan(np.zeros((1,1)),remaining_steps=1,budget=budget)
        self.assertEqual(inner.scores,0);self.assertEqual(budget.used,0.)

    def test_per_decision_cost_is_reset_but_legacy_totals_accumulate(self):
        from muslate.planning.cost import CostWeights
        c=CountingModel(Model(),weights=CostWeights(2.,3.,5.,.5))
        for _ in range(2):
            Lookahead(c,ProposalPolicy(c,n_candidates=4,mode='uniform'),depth=1).plan(np.zeros((1,1)),remaining_steps=1)
            self.assertEqual(c.last_cost.score_queries,1)
            self.assertEqual(c.last_cost.proposal_candidates,4)
            self.assertGreaterEqual(c.last_cost.planning_time_ms,c.last_cost.proposal_time_ms)
        self.assertEqual(c.n_score_calls,2)


if __name__=='__main__':unittest.main()
