import unittest
import numpy as np
from muslate.config import EnvConfig, CatalogConfig
from muslate.env.simulator import SlateSim
from ops.r2c_diagnostics import (make_bank, predict_case, discounted, spearman,
                                 summarize_case, bank_hash)


class HistorySensitiveModel:
    def encode_history(self, actions, rewards):
        return np.sum(rewards, axis=1)[:, None]

    def step(self, states, actions):
        return self.reward(states, actions), states + actions[:, :1] * .1 + .3

    def reward(self, states, actions):
        return states[:, 0] + actions.sum(axis=1) * .01


class R2CTests(unittest.TestCase):
    def setUp(self):
        self.sim = SlateSim(EnvConfig(catalog=CatalogConfig(n_coarse=2,n_fine=2,dim=4),
                                     slate_size=2,horizon=10,n_users=2))
        self.bank = make_bank(self.sim,1,2,n_sequences=7,probe_count=3)
        self.case = self.bank['cases'][0]
        self.model = HistorySensitiveModel()

    def test_horizon_one_modes_equal_and_same_probe(self):
        prediction = predict_case(self.model,self.case)
        np.testing.assert_array_equal(prediction['open_loop'][:,0],prediction['teacher_forced'][:,0])
        np.testing.assert_array_equal(prediction['probe_open_loop'][:,0],prediction['probe_teacher_forced'][:,0])

    def test_current_future_reward_perturbation_cannot_change_current_prediction(self):
        import copy
        original = predict_case(self.model,self.case)
        for t in range(5):
            changed = copy.deepcopy(self.case)
            rewards = np.array(changed['true_rewards']); rewards[:,t:] += 10000
            changed['true_rewards'] = rewards.tolist()
            prediction = predict_case(self.model,changed)
            np.testing.assert_array_equal(original['teacher_forced'][:,:t+1],prediction['teacher_forced'][:,:t+1])
            np.testing.assert_array_equal(original['open_loop'],prediction['open_loop'])
        changed = copy.deepcopy(self.case)
        rewards = np.array(changed['true_rewards']); rewards[:,0] += 100
        changed['true_rewards'] = rewards.tolist()
        self.assertFalse(np.array_equal(original['teacher_forced'][:,1],predict_case(self.model,changed)['teacher_forced'][:,1]))

    def test_bank_deterministic_legal_nested_and_checkpoint_independent(self):
        self.assertEqual(self.bank,make_bank(self.sim,1,2,n_sequences=7,probe_count=3))
        self.assertNotEqual(bank_hash(self.bank),bank_hash(make_bank(self.sim,2,2,n_sequences=7,probe_count=3)))
        for case in self.bank['cases']:
            sequences = np.array(case['sequences'])
            self.assertEqual(sequences.shape,(7,5,2))
            for a in sequences.reshape(-1,2):
                self.assertEqual(len(set(a)),2); self.assertTrue(np.all((a>=0)&(a<4)))
            for h in (1,3,5): np.testing.assert_array_equal(sequences[:,:h],sequences[:,:5][:,:h])
            probes=np.asarray(case['probes'])
            for j in range(1,7): np.testing.assert_array_equal(probes[0,0],probes[j,0])

    def test_discounting_and_rank_ties(self):
        np.testing.assert_allclose(discounted([[2.,4.,8.]],.5,3),[6.])
        self.assertAlmostEqual(spearman([1,1,3],[2,2,9]),1.)
        self.assertIsNone(spearman([1,1,1],[1,2,3]))

    def test_selected_sequence_is_open_loop_choice_in_both_modes(self):
        true = np.zeros((7,5)); ol=true.copy(); tf=true.copy()
        ol[2]=5; tf[4]=100; tf[2]=9
        probes=np.zeros((7,5,3))
        changed=dict(self.case,true_rewards=true.tolist())
        rows=summarize_case(changed,dict(open_loop=ol,teacher_forced=tf,
                    probe_open_loop=probes,probe_teacher_forced=probes),.95)
        for h in ('1','3','5'):
            row=rows['horizons'][h]
            self.assertEqual(row['selected_open_loop_index'],2)
            self.assertAlmostEqual(row['teacher_forced']['selected_overestimation'],discounted(tf,.95,int(h))[2])

    def test_model_identity_rejects_label_and_swapped_arm(self):
        import copy
        from ops.r2c_audit import validate_model_identity
        arm=dict(id='L0_s1_w0',condition='L0',seed=1,w_score=0,oracle_state=False,score_supervision='none')
        entry=dict(training_identity=dict(config=dict(arm=arm)))
        protocol=dict(expected_models=['L0_s1'],source_commit='a'*40,source={'code':'sha'},
                      checkpoints={'L0_s1':entry},seed_to_gpu={'1':3})
        row=dict(status='complete',condition='L0',seed=1,source_commit='a'*40,
                 identity=dict(source={'code':'sha'},protocol_file_sha256='protocol',checkpoint=entry,
                               training_identity=entry['training_identity'],bank={'sha':'bank'}),
                 runtime=dict(device='cuda:3',gpu='GPU',torch='t',numpy='n',python='p',executable='e',
                              command=['python','--seed','1','--device','cuda:3','--source-commit','a'*40]))
        validate_model_identity(row,'L0_s1',protocol,{'sha':'bank'},'protocol')
        changed=copy.deepcopy(row);changed['condition']='L1'
        with self.assertRaisesRegex(ValueError,'label/stem'): validate_model_identity(changed,'L0_s1',protocol,{'sha':'bank'},'protocol')
        # Even a coherently altered row+protocol cannot turn seed2 metadata into seed1.
        changed=copy.deepcopy(row);p=copy.deepcopy(protocol)
        changed['identity']['training_identity']['config']['arm']['seed']=2
        p['checkpoints']['L0_s1']=changed['identity']['checkpoint']
        with self.assertRaisesRegex(ValueError,'arm identity'): validate_model_identity(changed,'L0_s1',p,{'sha':'bank'},'protocol')

    def test_report_rejects_changed_verified_input(self):
        import tempfile,json
        from pathlib import Path
        from ops.r2c_run import digest
        from ops.r2c_report import verify_report_inputs
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'protocol.json';path.write_text('{}')
            (root/'verified_manifest.json').write_text(json.dumps(dict(status='verified',input_sha256={'protocol.json':digest(path)})))
            verify_report_inputs(root)
            path.write_text('{"tampered":true}')
            with self.assertRaisesRegex(ValueError,'verified input changed'): verify_report_inputs(root)


if __name__=='__main__': unittest.main()
