import sys,unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

class ReviewRegressionTests(unittest.TestCase):
    def test_full_selection_chain_binds_final_summary_to_actual_frozen_file(self):
        from tests.test_result_integrity import coherent_v2_results
        from scripts.reevaluate_stage1 import select_stage1,require_selection,stage_source,digest
        from scripts.analyze_logged_pilot import build_summary
        results=coherent_v2_results();source=stage_source()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for r in results:
                r['identity']['source']=source;arm=r['identity']['config']['arm']
                p=root/arm['id']/'validation.json';p.parent.mkdir()
                p.write_text(json.dumps(dict(identity=r['identity'],protocols=r['protocols']['validation'])))
            path=root/'selection.json';select_stage1(root,path)
            frozen=require_selection(path,source);sha=digest(path)
            for r in results:r['metadata']['selection_sha256']=sha
            summary=build_summary(results,frozen_selection=frozen,frozen_selection_sha256=sha)
            self.assertEqual(summary['selection'],{'L0':0,'L1':100,'U1':100})
            results[0]['metadata']['selection_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'another frozen selection'):
                build_summary(results,frozen_selection=frozen,frozen_selection_sha256=sha)

    def test_duplicate_pinned_validation_files_are_rejected(self):
        from scripts.reevaluate_stage1 import require_selection,digest
        from scripts.analyze_logged_pilot import SELECTION_RULE
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'validation.json';f.write_text('{}')
            s=dict(source={},selection_rule=SELECTION_RULE,selection={'L0':0,'L1':100,'U1':100},
                   validation_files=[dict(path=str(f),sha256=digest(f))]*21)
            p=Path(d)/'selection.json';p.write_text(json.dumps(s))
            with self.assertRaises(ValueError):require_selection(p,{})

    def test_pinned_raw_validation_must_reproduce_selected_weights(self):
        from scripts.reevaluate_stage1 import require_selection,digest
        from scripts.run_logged_pilot import build_pilot_arms
        from scripts.analyze_logged_pilot import SELECTION_RULE
        with tempfile.TemporaryDirectory() as d:
            entries=[]
            for arm in build_pilot_arms((1,2,3)):
                f=Path(d)/arm['id']/'validation.json';f.parent.mkdir()
                f.write_text(json.dumps(dict(identity=dict(config=dict(arm=arm,protocol_version=2),source={}),protocols={})))
                entries.append(dict(path=str(f),sha256=digest(f)))
            s=dict(source={},contract={'protocol_version':2},seeds=[1,2,3],selection_rule=SELECTION_RULE,
                   selection={'L0':0,'L1':1,'U1':1},validation_files=entries)
            p=Path(d)/'selection.json';p.write_text(json.dumps(s))
            with patch('scripts.reevaluate_stage1.validate_results',return_value=({},s['contract'])),patch(
                    'scripts.reevaluate_stage1.select_weights',return_value={'L0':0,'L1':100,'U1':100}):
                with self.assertRaises(ValueError):require_selection(p,{})

    def test_stage1_seed_range_cannot_overlap_splits(self):
        from scripts.reevaluate_stage1 import fresh_seed
        with self.assertRaises(ValueError):fresh_seed(5001,'validation')

    def test_capacity_sampling_recomputes_softmax_after_extreme_group_taken(self):
        from muslate.planning.abstractions import SIDProposal
        class Model:n_items=5;slate_size=5
        p=SIDProposal(Model(),5,1,n_sampled=8)
        plans=p._coarse_plans(np.array([1000.,0.,0.,0.,0.]),5,8)
        self.assertTrue(all(len(set(a))==5 for a in plans))

    def test_queue_contract_can_resume_and_rejects_changed_source(self):
        from ops.stage1_queue import claim_queue
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract={'phase':'validation','source':{'a':'hash'}}
            claim_queue(root,contract);claim_queue(root,contract)
            with self.assertRaises(ValueError):claim_queue(root,dict(contract,source={'a':'changed'}))

if __name__=='__main__':unittest.main()
