import unittest
import copy
import tempfile
from pathlib import Path
import numpy as np
from ops.r2b_search import Search,METHODS,labels_for,validate_ledger
from ops.r2b_identity import require_commit,validate_command,atomic,GPU_SLOTS,gpu_for_seed

class Model:
    n_items=12;slate_size=3
    def item_scores(self,states):return np.tile(np.arange(12,dtype=float),(len(states),1))
    def reward(self,states,actions):
        return np.array([float(a[0]==2)*50+sum(a)/100 for a in actions])

def calibration():
    return dict(reward_ms={str(n):1+.1*n for n in range(1,33)},score_ms=1.,setup_ms=.2,
                uniform_ms=.1,sid_ms=.2,refine_ms=.1,max_batch=32)

class R2BTests(unittest.TestCase):
    def test_frozen_gpu_assignment(self):
        self.assertEqual(GPU_SLOTS,(2,4,5))
        self.assertEqual([gpu_for_seed(s) for s in range(1,6)],[2,4,5,2,4])
        for s in (0,6):
            with self.assertRaises(ValueError):gpu_for_seed(s)
    def setUp(self):self.model=Model();self.state=np.zeros((1,4));self.labels=np.repeat(np.arange(3),4)
    def test_legal_and_preflight_boundary_all_methods(self):
        c=calibration();minimum=c['score_ms']+c['setup_ms']+c['reward_ms']['1']
        for method in METHODS:
            with self.assertRaises(ValueError):Search(self.model,c,minimum-1e-6,1,self.labels).plan(self.state,method)
            for cap in (minimum,minimum+.15,7.,40.):
                a,row=Search(self.model,c,cap,1,self.labels).plan(self.state,method)
                self.assertEqual(len(set(a)),3);self.assertTrue(np.all((a>=0)&(a<12)))
                self.assertLessEqual(row['estimated_ms'],cap+1e-9)
                self.assertTrue(all(x<=y for x,y in zip(row['incumbent_trace'],row['incumbent_trace'][1:])))
                self.assertAlmostEqual(sum(e['estimated_ms'] for e in row['events']),row['estimated_ms'])
                self.assertEqual(row['error'],None)
                validate_ledger(row,dict(coefficients=c,mid_cap_ms=cap))
    def test_shared_anchor_replay_and_capacity(self):
        anchors=[]
        for method in METHODS:
            a,row=Search(self.model,calibration(),25.,99,self.labels).plan(self.state,method)
            a2,row2=Search(self.model,calibration(),25.,99,self.labels).plan(self.state,method)
            np.testing.assert_array_equal(a,a2);self.assertEqual(row,row2);anchors.append(row['anchor'])
        self.assertTrue(all(a==anchors[0] for a in anchors))
        for seed in (271,811,1459):
            np.testing.assert_array_equal(np.bincount(labels_for(self.labels,seed)),np.bincount(self.labels))
    def test_unrestricted_scan_reaches_beyond_top_three(self):
        a,row=Search(self.model,calibration(),100.,11,self.labels).plan(self.state,'flat_coordinate_refine',diagnostic=True)
        self.assertEqual(a[0],2);self.assertGreater(row['predicted_reward'],row['anchor_prediction'])
    def test_error_returns_evaluated_anchor(self):
        class Broken(Model):
            calls=0
            def reward(self,states,actions):
                self.calls+=1
                if self.calls>1:raise RuntimeError('fixture failure')
                return super().reward(states,actions)
        a,row=Search(Broken(),calibration(),100.,1,self.labels).plan(self.state,'uniform')
        self.assertEqual(a.tolist(),row['anchor']);self.assertEqual(row['stop_reason'],'error_fallback')
        self.assertIn('fixture failure',row['error'])
    def test_refinement_slots_prepaid_even_when_invalid(self):
        c=calibration();minimum=c['score_ms']+c['setup_ms']+c['reward_ms']['1']
        _,empty=Search(self.model,c,minimum+.01,11,self.labels).plan(self.state,'flat_coordinate_refine',diagnostic=True)
        self.assertEqual(empty['cost'].get('refinement_attempts',0),0)
        _,row=Search(self.model,c,100.,11,self.labels).plan(self.state,'flat_coordinate_refine',diagnostic=True)
        self.assertEqual(row['cost']['refinement_attempts'],row['cost']['invalid_refinement_slots']+row['cost']['legal_refinement_candidates'])
        self.assertGreater(row['cost']['invalid_refinement_slots'],0)
    def test_audit_rejects_cap_event_and_counter_tampering(self):
        c=calibration();_,row=Search(self.model,c,40.,7,self.labels).plan(self.state,'flat_coordinate_refine')
        mutations=(lambda x:x.update(cap_ms=41.),lambda x:x['events'][0].update(estimated_ms=.5),
                   lambda x:x['cost'].update(reward_samples=999),lambda x:x.update(error='unexpected'))
        for mutate in mutations:
            altered=copy.deepcopy(row);mutate(altered)
            with self.assertRaises(ValueError):validate_ledger(altered,dict(coefficients=c,mid_cap_ms=40.))
    def test_launch_command_device_and_commit_tampering(self):
        commit='a'*40;command=['python','ops/r2b_run.py','--device','cuda:1','--seed','2','--source-commit',commit]
        expected={'--device':'cuda:1','--seed':2,'--source-commit':commit};validate_command(command,expected)
        for index,value in ((3,'cuda:2'),(5,'3'),(7,'abc')):
            changed=command.copy();changed[index]=value
            with self.assertRaises(ValueError):validate_command(changed,expected)
        for value in (None,'HEAD','a'*39,'z'*40):
            with self.assertRaises(ValueError):require_commit(value)
    def test_atomic_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cell.json';atomic(path,dict(original=1));before=path.read_bytes()
            with self.assertRaises(FileExistsError):atomic(path,dict(replaced=2))
            self.assertEqual(path.read_bytes(),before);self.assertEqual(list(Path(directory).glob('*.tmp')),[])
if __name__=='__main__':unittest.main()
