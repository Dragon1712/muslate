import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tests import test_pilot_analysis as fixtures
from scripts.analyze_logged_pilot import validate_results,select_weights


def coherent_v2_results():
    from muslate.planning.cost import PlanningCost
    from dataclasses import asdict
    results=copy.deepcopy(fixtures.AnalysisTests().fixture())
    for r in results:
        settings=r['identity']['config'];settings['protocol_version']=2
        h=settings['environment']['horizon'];gamma=settings['environment']['gamma']
        for split in r['protocols'].values():
            for p in split.values():
                warm=p['warmup'];discount=sum(gamma**t for t in range(warm,h))
                for planner,row in p['rows'].items():
                    row['decisions']=[dict(user_id=u,t=t,remaining_steps=h-t,
                        effective_depth=0 if planner=='reactive' else min(settings['depth'],h-t-1),
                        slate=[0,1,2,3,4],reward=row['tail_returns'][i]/discount,
                        cost=asdict(PlanningCost()),estimated_compute_ms=0.) for i,u in enumerate(p['user_ids']) for t in range(warm,h)]
                    row.update(cost_weights=dict(reward_ms=2.,transition_ms=3.,score_ms=5.,proposal_ms=.5),
                        cost_totals=asdict(PlanningCost()),estimated_compute_ms=0.,score_calls=0,slate_evals=0)
    return results


class ResultIntegrityTests(unittest.TestCase):
    def test_version2_rejects_illegal_recorded_action_and_future_depth(self):
        from muslate.eval.integrity import validate_protocols
        r=coherent_v2_results()[0];settings=r['identity']['config']
        validate_protocols(r['protocols'],settings)
        for mutation,message in (('slate','duplicated item'),('depth','finite horizon')):
            changed=copy.deepcopy(r)
            record=changed['protocols']['validation']['full_episode']['rows']['sid_hier']['decisions'][-1]
            if mutation=='slate':record['slate']=[0,0,2,3,4]
            else:record['effective_depth']=2
            with self.assertRaisesRegex(ValueError,message):validate_protocols(changed['protocols'],settings)

    def test_incomplete_completed_run_is_not_silently_skipped(self):
        import tempfile,json
        from scripts.run_logged_pilot import claim_run
        identity={'config':{'epochs':1},'source':{'x':'y'}}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);claim_run(root,identity)
            (root/'result.json').write_text(json.dumps({'identity':identity,'status':'complete'}))
            with self.assertRaises(ValueError):claim_run(root,identity)

    def test_mean_mismatch_is_rejected_before_validation_selection(self):
        results=copy.deepcopy(fixtures.AnalysisTests().fixture())
        results[1]['protocols']['validation']['common_prefix']['rows']['sid_hier']['return_mean']=1e100
        with self.assertRaises(ValueError):validate_results(results)
        with self.assertRaises(ValueError):select_weights(results)

    def test_tail_must_sum_with_prefix_to_total_return(self):
        results=copy.deepcopy(fixtures.AnalysisTests().fixture())
        row=results[0]['protocols']['test']['common_prefix']['rows']['sid_hier']
        row['tail_returns']=[-999.,-999.]
        with self.assertRaises(ValueError):validate_results(results)

    def test_missing_planner_and_split_fail_explicitly(self):
        for missing in ('planner','split'):
            results=copy.deepcopy(fixtures.AnalysisTests().fixture())
            if missing=='split':del results[0]['protocols']['test']
            else:del results[0]['protocols']['validation']['common_prefix']['rows']['reactive']
            with self.assertRaises(ValueError):validate_results(results)


if __name__=='__main__':unittest.main()
