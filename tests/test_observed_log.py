"""Observed data must never export simulator state or unexposed labels."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muslate.config import EnvConfig, LogConfig
from muslate.env.simulator import SlateSim
from muslate.env import logger


class ObservedLogTests(unittest.TestCase):
    def setUp(self):
        self.sim = SlateSim(EnvConfig(horizon=3, long_term_strength=3, div_penalty=.8))
        self.cfg = LogConfig(n_episodes=2, seed=7)

    def test_observed_log_has_no_state_channel(self):
        self.assertTrue(hasattr(logger, 'generate_observed_log'), 'missing observable export')
        log = logger.generate_observed_log(self.sim, self.cfg)
        self.assertFalse(hasattr(log, 'states'))
        self.assertEqual(log.item_feedback.shape, (6, 5))
        self.assertTrue(np.isfinite(log.item_feedback).all())

    def test_feedback_rng_does_not_change_logged_actions_or_rewards(self):
        self.assertTrue(hasattr(logger, 'generate_observed_log'))
        old = logger.generate_log(self.sim, self.cfg)
        new = logger.generate_observed_log(self.sim, self.cfg)
        np.testing.assert_array_equal(old.slates, new.slates)
        np.testing.assert_array_equal(old.rewards, new.rewards)

    def test_exposed_feedback_matches_unary_without_noise(self):
        self.assertTrue(hasattr(logger, 'generate_observed_log'))
        cfg = LogConfig(n_episodes=2, seed=7, item_feedback_noise=0)
        old = logger.generate_log(self.sim, cfg)
        new = logger.generate_observed_log(self.sim, cfg)
        for row in (0, 2, 4):
            h = old.states[row, :self.sim.cfg.dim]
            sat = old.states[row, self.sim.cfg.dim:]
            ids = old.slates[row]
            expected = self.sim.catalog.item_emb[ids] @ h - 1.5 * sat[self.sim.catalog.item_coarse[ids]]
            np.testing.assert_allclose(new.item_feedback[row], expected, atol=1e-12)

    def test_round_trip_preserves_zero_and_missing_feedback(self):
        self.assertTrue(hasattr(logger, 'ObservedLog'))
        log = logger.generate_observed_log(self.sim, self.cfg)
        log.item_feedback[0, 0] = 0
        log.item_feedback[0, 1] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'observed.npz'
            log.save(path)
            with np.load(path, allow_pickle=False) as data:
                self.assertNotIn('states', data.files)
            restored = logger.ObservedLog.load(path)
        self.assertEqual(restored.item_feedback[0, 0], 0)
        self.assertTrue(np.isnan(restored.item_feedback[0, 1]))
        np.testing.assert_array_equal(restored.slates, log.slates)

    def test_reward_only_conversion_has_no_feedback_or_aliases(self):
        self.assertTrue(hasattr(logger, 'as_observed_log'))
        old = logger.generate_log(self.sim, self.cfg)
        observed = logger.as_observed_log(old)
        self.assertIsNone(observed.item_feedback)
        self.assertFalse(hasattr(observed, 'states'))
        observed.slates[0, 0] = (old.slates[0, 0] + 1) % self.sim.n_items
        self.assertNotEqual(observed.slates[0, 0], old.slates[0, 0])


if __name__ == '__main__':
    unittest.main()
