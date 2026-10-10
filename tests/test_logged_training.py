"""Training must respect the observable information boundary and missing labels."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muslate.config import EnvConfig, LogConfig, WorldModelConfig
from muslate.env.simulator import SlateSim
from muslate.env.logger import generate_observed_log
from muslate.model.world_model import WorldModel
from muslate.model import train

torch.set_num_threads(1)


class LoggedTrainingTests(unittest.TestCase):
    def fixture(self, n=2):
        sim = SlateSim(EnvConfig(horizon=2))
        log = generate_observed_log(sim, LogConfig(n_episodes=n, seed=11))
        cfg = WorldModelConfig(oracle_state=False, epochs=1, batch_size=2,
                               score_supervision='exposed', seed=5)
        return log, cfg

    def test_strict_training_needs_no_simulator_or_states(self):
        self.assertTrue(hasattr(train, 'train_logged_world_model'))
        log, cfg = self.fixture()
        model = train.train_logged_world_model(log, cfg, torch.device('cpu'), verbose=False)
        self.assertIsNone(model.state_head)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertFalse(hasattr(log, 'states'))

    def test_strict_training_rejects_privileged_config(self):
        self.assertTrue(hasattr(train, 'train_logged_world_model'))
        for state, mode in ((True, 'exposed'), (False, 'oracle_full')):
            log, cfg = self.fixture()
            cfg.oracle_state, cfg.score_supervision = state, mode
            with self.assertRaises(ValueError):
                train.train_logged_world_model(log, cfg, torch.device('cpu'), False)

    def test_exposed_loss_uses_only_finite_positions(self):
        self.assertTrue(hasattr(train, '_exposed_feedback_loss'))
        cfg = WorldModelConfig(oracle_state=False, pairwise=False)
        model = WorldModel(cfg, 4, 2)
        with torch.no_grad():
            for p in model.parameters(): p.zero_()
            model.q_head[-1].bias.copy_(torch.tensor([1., 2., 3., 4.]))
        h = torch.zeros(2, cfg.latent_dim)
        ids = torch.tensor([[0, 2], [1, 3]])
        target = torch.tensor([[0., float('nan')], [0., float('nan')]])
        loss = train._exposed_feedback_loss(model, h, ids, target)
        self.assertAlmostEqual(float(loss.detach()), 2.5)
        loss.backward()
        np.testing.assert_allclose(model.q_head[-1].bias.grad.numpy(), [1, 2, 0, 0])

    def test_all_missing_feedback_has_zero_finite_gradients(self):
        self.assertTrue(hasattr(train, '_exposed_feedback_loss'))
        model = WorldModel(WorldModelConfig(oracle_state=False), 4, 2)
        loss = train._exposed_feedback_loss(model, torch.zeros(1, 64),
                                          torch.tensor([[0, 1]]), torch.full((1, 2), float('nan')))
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_strict_training_never_calls_true_item_scores(self):
        self.assertTrue(hasattr(train, 'train_logged_world_model'))
        log, cfg = self.fixture()
        with patch.object(train, 'true_item_scores', side_effect=AssertionError('privileged read')):
            for mode in ('none', 'exposed'):
                cfg.score_supervision = mode
                train.train_logged_world_model(log, cfg, torch.device('cpu'), False)

    def test_small_and_partial_batches_are_trained(self):
        self.assertTrue(hasattr(train, 'train_logged_world_model'))
        original = WorldModel.forward
        for n, expected in ((1, [1]), (3, [2, 1])):
            sizes = []
            def record(model, slates, rewards):
                sizes.append(slates.shape[0])
                return original(model, slates, rewards)
            log, cfg = self.fixture(n)
            torch.manual_seed(cfg.seed)
            initial = WorldModel(cfg, log.n_items, log.slate_size)
            with patch.object(WorldModel, 'forward', record):
                fitted = train.train_logged_world_model(log, cfg, torch.device('cpu'), False)
            self.assertEqual(sizes, expected)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial.parameters(), fitted.parameters())))

    def test_current_and_future_feedback_do_not_change_current_belief(self):
        model = WorldModel(WorldModelConfig(oracle_state=False), 10, 2)
        ids = torch.tensor([[[0, 1], [2, 3], [4, 5]]])
        reward = torch.tensor([[1., 2., 3.]])
        altered = torch.tensor([[1., -200., 900.]])
        torch.testing.assert_close(model(ids, reward)[:, :2], model(ids, altered)[:, :2])

    def test_incomplete_or_reordered_episodes_are_rejected(self):
        self.assertTrue(hasattr(train, 'train_logged_world_model'))
        log, cfg = self.fixture()
        log.step_ids[[0, 1]] = log.step_ids[[1, 0]]
        with self.assertRaises(ValueError):
            train.train_logged_world_model(log, cfg, torch.device('cpu'), False)


if __name__ == '__main__': unittest.main()
