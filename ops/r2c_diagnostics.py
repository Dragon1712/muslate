"""Checkpoint-independent R2-C banks and chronology-explicit prediction diagnostics."""
import hashlib
import json
import numpy as np
from muslate.eval.rollout import build_warm_start_cases

HORIZONS = (1,3,5)
PREFIX_BASE = 31000000


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def bank_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def address_seed(*parts):
    return int.from_bytes(hashlib.sha256(canonical(parts)).digest()[:8],'little')


def state_record(state):
    return dict(h=state.h.tolist(),sat=state.sat.tolist(),t=int(state.t),
                slates=[np.asarray(a).tolist() for a in state.slates],rewards=list(map(float,state.rewards)))


def uniform_slates(rng,n,n_items,slate_size):
    return np.stack([rng.choice(n_items,slate_size,replace=False) for _ in range(n)])


def make_bank(sim,seed,n_users,n_sequences=64,probe_count=16):
    """Generate all actions before prediction; model/condition are absent by construction."""
    cases=build_warm_start_cases(sim,n_users,5,PREFIX_BASE+2000*seed)
    rows=[]
    for case in cases:
        rng=np.random.default_rng(address_seed('r2c_sequences',seed,case.user_id))
        sequences=uniform_slates(rng,n_sequences*5,sim.n_items,sim.slate_size).reshape(n_sequences,5,sim.slate_size)
        truths=[]; probes=[]; probe_truths=[]; reached=[]
        for j,sequence in enumerate(sequences):
            state=case.state.clone();rs=[];ps=[];pts=[];identities=[]
            for t,action in enumerate(sequence):
                identities.append(bank_hash(state_record(state)))
                prng=np.random.default_rng(address_seed('r2c_probe',seed,case.user_id,identities[-1]))
                p=uniform_slates(prng,probe_count,sim.n_items,sim.slate_size)
                # A simulator step clones its input. Explicit clones keep this invariant visible.
                pt=[float(sim.step(state.clone(),a,noise=False)[0]) for a in p]
                r,state=sim.step(state,action,noise=False)
                rs.append(float(r));ps.append(p.tolist());pts.append(pt)
            truths.append(rs);probes.append(ps);probe_truths.append(pts);reached.append(identities)
        initial=state_record(case.state)
        history=dict(slates=[a.tolist() for a in case.hist_slates],rewards=case.hist_rewards)
        rows.append(dict(user_id=case.user_id,episode_seed=case.episode_seed,prefix_return=case.prefix_return,
                         initial_state=initial,initial_state_sha256=bank_hash(initial),history=history,
                         history_sha256=bank_hash(history),sequences=sequences.tolist(),true_rewards=truths,
                         probes=probes,probe_true_rewards=probe_truths,reached_state_sha256=reached))
    return dict(format_version=1,seed=seed,prefix_seed=PREFIX_BASE+2000*seed,n_users=n_users,
                n_sequences=n_sequences,sequence_length=5,probe_count=probe_count,cases=rows)


def discounted(rewards,gamma,horizon):
    r=np.asarray(rewards,dtype=np.float64)
    return r[:,:horizon] @ (gamma**np.arange(horizon))


def average_ranks(values):
    a=np.asarray(values,dtype=np.float64);order=np.argsort(a,kind='stable');r=np.empty(len(a),dtype=np.float64)
    start=0
    while start<len(a):
        end=start+1
        while end<len(a) and a[order[end]]==a[order[start]]: end+=1
        r[order[start:end]]=(start+end-1)/2.+1
        start=end
    return r


def spearman(x,y):
    x=average_ranks(x);y=average_ranks(y);x-=x.mean();y-=y.mean()
    denom=float(np.linalg.norm(x)*np.linalg.norm(y))
    return None if denom==0 else float(np.dot(x,y)/denom)


def predict_case(model,case):
    sequences=np.asarray(case['sequences'],dtype=np.int64);true=np.asarray(case['true_rewards'])
    probes=np.asarray(case['probes'],dtype=np.int64);n,tmax,k=sequences.shape
    prefix=np.repeat(np.asarray(case['history']['slates'],dtype=np.int64)[None],n,axis=0)
    prefix_rewards=np.repeat(np.asarray(case['history']['rewards'])[None],n,axis=0)
    # This is the only history consumed by the open-loop branch.
    latent=model.encode_history(prefix,prefix_rewards)
    ol=[];tf=[];pol=[];ptf=[]
    for t in range(tmax):
        # Slice strictly preceding observations before invoking the encoder.
        teacher=model.encode_history(np.concatenate((prefix,sequences[:,:t]),axis=1),
                                     np.concatenate((prefix_rewards,true[:,:t]),axis=1))
        probe_actions=probes[:,t].reshape(-1,k);count=probes.shape[2]
        pol.append(model.reward(np.repeat(latent,count,axis=0),probe_actions).reshape(n,count))
        ptf.append(model.reward(np.repeat(teacher,count,axis=0),probe_actions).reshape(n,count))
        r,latent=model.step(latent,sequences[:,t]);ol.append(r)
        tf.append(model.reward(teacher,sequences[:,t]))
    result=dict(open_loop=np.stack(ol,axis=1),teacher_forced=np.stack(tf,axis=1),
                probe_open_loop=np.stack(pol,axis=1),probe_teacher_forced=np.stack(ptf,axis=1))
    for value in result.values():
        if not np.isfinite(value).all(): raise ValueError('nonfinite prediction')
    if not np.array_equal(result['open_loop'][:,0],result['teacher_forced'][:,0]):
        raise ValueError('horizon1 mode mismatch')
    return result


def summarize_case(case,prediction,gamma):
    truth=np.asarray(case['true_rewards']);probe_truth=np.asarray(case['probe_true_rewards'])
    # Normalize the reduction dtype so metrics replay identically from JSON floats.
    ol=np.asarray(prediction['open_loop'],dtype=np.float64);tf=np.asarray(prediction['teacher_forced'],dtype=np.float64)
    horizons={}
    for h in HORIZONS:
        tr=discounted(truth,gamma,h);ov=discounted(ol,gamma,h);tv=discounted(tf,gamma,h)
        selected=int(np.argmax(ov));row=dict(selected_open_loop_index=selected)
        for name,pred,value in [('open_loop',ol,ov),('teacher_forced',tf,tv)]:
            row[name]=dict(reward_mae=float(np.abs(pred[:,:h]-truth[:,:h]).mean()),
                           return_mae=float(np.abs(value-tr).mean()),rank_correlation=spearman(value,tr),
                           selected_prediction=float(value[selected]),selected_truth=float(tr[selected]),
                           selected_overestimation=float(value[selected]-tr[selected]))
        horizons[str(h)]=row
    po=np.asarray(prediction['probe_open_loop'],dtype=np.float64);pt=np.asarray(prediction['probe_teacher_forced'],dtype=np.float64)
    return dict(horizons=horizons,
                reward_mae_by_step={name:np.abs(np.asarray(prediction[name])-truth).mean(axis=0).tolist()
                                    for name in ('open_loop','teacher_forced')},
                probe=dict(functional_difference_by_step=np.abs(po-pt).mean(axis=(0,2)).tolist(),
                           open_loop_mae_by_step=np.abs(po-probe_truth).mean(axis=(0,2)).tolist(),
                           teacher_forced_mae_by_step=np.abs(pt-probe_truth).mean(axis=(0,2)).tolist()))


def aggregate(cases):
    horizons={}
    for h in map(str,HORIZONS):
        row={}
        for mode in ('open_loop','teacher_forced'):
            metrics=[c['metrics']['horizons'][h][mode] for c in cases]
            row[mode]={key:float(np.mean([m[key] for m in metrics]))
                       for key in ('reward_mae','return_mae','selected_overestimation')}
            ranks=[m['rank_correlation'] for m in metrics if m['rank_correlation'] is not None]
            row[mode].update(rank_correlation_mean=float(np.mean(ranks)) if ranks else None,
                             rank_defined=len(ranks),rank_undefined=len(metrics)-len(ranks))
        horizons[h]=row
    return dict(horizons=horizons,
                reward_mae_by_step={mode:np.mean([c['metrics']['reward_mae_by_step'][mode] for c in cases],axis=0).tolist()
                                    for mode in ('open_loop','teacher_forced')},
                probe={key:np.mean([c['metrics']['probe'][key] for c in cases],axis=0).tolist()
                       for key in cases[0]['metrics']['probe']})
