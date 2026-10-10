"""Result schema validation shared by analysis and stage resumption."""
import numpy as np
from muslate.planning.validity import validate_slates
from muslate.planning.cost import PlanningCost,CostWeights
from dataclasses import asdict

PLANNERS=('reactive','flat_uniform','flat_policy','sid_flat','sid_hier')


def checked_returns(row,n=None):
    try:
        a=np.asarray(row['returns'],dtype=float)
        mean=float(row['return_mean'])
    except (KeyError,TypeError,ValueError) as exc:raise ValueError('missing/invalid returns') from exc
    if a.ndim!=1 or not len(a) or (n is not None and len(a)!=n) or not np.isfinite(a).all():
        raise ValueError('invalid return vector')
    if not np.isfinite(mean) or not np.isclose(mean,a.mean(),rtol=1e-9,atol=1e-9):
        raise ValueError('inconsistent return_mean')
    return a


def validate_protocols(protocols,settings,splits=('validation','test')):
    try:
        n=settings['users'];h=settings['environment']['horizon']
        for split in splits:
            for name in ('full_episode','common_prefix'):
                p=protocols[split][name]
                warm=0 if name=='full_episode' else settings['warmup']
                if p['warmup']!=warm or not 0<=warm<h:raise ValueError('invalid warmup')
                ids=p['user_ids'];prefix=np.asarray(p['prefix_returns'],dtype=float)
                seeds=p['episode_seeds']
                if (len(ids)!=n or len(set(ids))!=n or len(seeds)!=n or prefix.shape!=(n,)
                        or not np.isfinite(prefix).all() or (warm==0 and np.any(prefix!=0))):
                    raise ValueError('invalid case identities/prefix')
                if set(p['rows'])!=set(PLANNERS):raise ValueError('missing/unknown planner')
                for planner,row in p['rows'].items():
                    a=checked_returns(row,n);tail=np.asarray(row['tail_returns'],dtype=float)
                    if tail.shape!=(n,) or not np.isfinite(tail).all() or not np.allclose(a,prefix+tail,rtol=1e-9,atol=1e-9):
                        raise ValueError('inconsistent prefix/tail returns')
                    if not np.isclose(float(row['tail_mean']),tail.mean(),rtol=1e-9,atol=1e-9):
                        raise ValueError('inconsistent tail mean')
                    if row['n_decisions']!=n*(h-warm):raise ValueError('invalid decision count')
                    if settings.get('protocol_version',1)>=2:
                        records=row['decisions']
                        if len(records)!=row['n_decisions']:raise ValueError('missing per-decision evidence')
                        expected={(u,t) for u in ids for t in range(warm,h)};seen=set()
                        totals={k:0. for k in asdict(PlanningCost())};returns={u:0. for u in ids}
                        catalog=settings['environment']['catalog'];count=catalog['n_coarse']*catalog['n_fine']
                        compute=0.;weights=CostWeights(**row['cost_weights'])
                        for record in records:
                            key=(record['user_id'],record['t'])
                            if key not in expected or key in seen:raise ValueError('invalid decision identity')
                            seen.add(key);remaining=h-record['t']
                            depth=0 if planner=='reactive' else min(settings['depth'],remaining-1)
                            if record['remaining_steps']!=remaining or record['effective_depth']!=depth:
                                raise ValueError('invalid finite horizon evidence')
                            validate_slates(record['slate'],count,settings['environment']['slate_size'])
                            reward=float(record['reward'])
                            if not np.isfinite(reward):raise ValueError('invalid observed reward')
                            returns[key[0]]+=settings['environment']['gamma']**key[1]*reward
                            cost=PlanningCost(**record['cost'])
                            for k,v in asdict(cost).items():
                                if not np.isfinite(v) or v<0:raise ValueError('invalid cost evidence')
                                totals[k]+=v
                            estimate=cost.estimate(weights)
                            if not np.isclose(estimate,record['estimated_compute_ms'],rtol=1e-9,atol=1e-9):
                                raise ValueError('inconsistent compute cost')
                            compute+=estimate
                        if not np.allclose([returns[u] for u in ids],tail,rtol=1e-9,atol=1e-9):
                            raise ValueError('observed rewards conflict with return')
                        if any(not np.isclose(v,row['cost_totals'][k],rtol=1e-9,atol=1e-9) for k,v in totals.items()):
                            raise ValueError('inconsistent aggregated cost')
                        if not np.isclose(compute,row['estimated_compute_ms'],rtol=1e-9,atol=1e-9):
                            raise ValueError('inconsistent compute sum')
                        if (row['score_calls']!=totals['score_queries']
                                or row['slate_evals']!=totals['reward_only_calls']+totals['transition_calls']):
                            raise ValueError('inconsistent operation counters')
    except (KeyError,TypeError) as exc:raise ValueError('incomplete result schema') from exc
