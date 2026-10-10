"""Common-prefix comparisons must clone histories and preserve episode discount."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muslate.config import EnvConfig
from muslate.env.simulator import SlateSim
from muslate.eval import rollout


class FixedPlanner:
    def plan(self, latent, remaining_steps=None, budget=None): return np.asarray([[0, 1, 2, 3, 4]])


class WarmStartTests(unittest.TestCase):
    def fixture(self): return SlateSim(EnvConfig(horizon=4, gamma=.5))

    def test_planners_receive_identical_cloned_prefixes(self):
        self.assertTrue(hasattr(rollout, 'build_warm_start_cases'))
        sim = self.fixture()
        cases = rollout.build_warm_start_cases(sim, 1, warmup=2, seed=8)
        saved = cases[0].hist_slates[0].copy()
        first = []
        def inspect(state, hs, hr):
            if state.t == 2:
                first.append((state.h.copy(), np.stack(hs).copy(), list(hr)))
                hs[0][0] = (hs[0][0] + 1) % sim.n_items
            return np.zeros((1, 1))
        for _ in range(2):
            rollout.evaluate_from_prefixes(sim, FixedPlanner(), inspect, cases, False)
        for a, b in zip(first[0], first[1]): np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(cases[0].hist_slates[0], saved)

    def test_suffix_discount_matches_full_episode_clock(self):
        self.assertTrue(hasattr(rollout, 'build_warm_start_cases'))
        sim = self.fixture()
        cases = rollout.build_warm_start_cases(sim, 1, warmup=2)
        # After the common prefix, both remaining rewards are exactly 2.
        sim.reward_batch = lambda h, sat, slates: np.full(len(slates), 2.)
        result = rollout.evaluate_from_prefixes(sim, FixedPlanner(), lambda *a: np.zeros((1, 1)), cases, False)
        self.assertAlmostEqual(result.tail_returns[0], .75)  # 2*.5**2 + 2*.5**3
        self.assertAlmostEqual(result.returns[0], cases[0].prefix_return + .75)
        self.assertEqual(result.n_decisions, 2)

    def test_zero_warmup_matches_existing_evaluate(self):
        self.assertTrue(hasattr(rollout, 'build_warm_start_cases'))
        sim = self.fixture()
        latent = lambda *a: np.zeros((1, 1))
        expected = rollout.evaluate(sim, FixedPlanner(), latent, 2, seed=8, measure_latency=False)
        cases = rollout.build_warm_start_cases(sim, 2, warmup=0, seed=8)
        actual = rollout.evaluate_from_prefixes(sim, FixedPlanner(), latent, cases, False)
        np.testing.assert_allclose(actual.returns, expected.returns)

    def test_invalid_warmup_is_rejected(self):
        self.assertTrue(hasattr(rollout, 'build_warm_start_cases'))
        for warmup in (-1, 4):
            with self.assertRaises(ValueError):
                rollout.build_warm_start_cases(self.fixture(), 1, warmup=warmup)


if __name__ == '__main__': unittest.main()

