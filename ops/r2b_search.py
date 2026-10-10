"""Runner-local, prepaid depth-zero search. No changes to frozen planners."""
from collections import Counter
import numpy as np

RANDOM_SEEDS = (271, 811, 1459)
METHODS = ('uniform', 'flat_coordinate_refine', 'true_sid_flat', 'true_sid_hier') + tuple(
    f'random_sid_{mode}_{seed}' for mode in ('flat', 'hier') for seed in RANDOM_SEEDS)


def labels_for(reference, seed=None):
    reference = np.asarray(reference, dtype=np.int64)
    return reference.copy() if seed is None else np.random.default_rng(seed).permutation(reference)


def prepare(scores, labels):
    order = np.argsort(-scores, kind='stable')
    groups = [order[labels[order] == c] for c in np.unique(labels)]
    coarse = np.array([scores[g[0]] for g in groups])
    p = np.exp((coarse-coarse.max())/.7); p /= p.sum()
    return order, groups, p


def generate(rng, n_items, k, groups=None, probabilities=None):
    if groups is None:
        return rng.choice(n_items, k, replace=False).astype(np.int64)
    used = set(); result = []
    for _ in range(k):
        available = [c for c,g in enumerate(groups) if any(int(i) not in used for i in g)]
        p = probabilities[available]; p = p/p.sum()
        c = int(rng.choice(available, p=p))
        item = next(int(i) for i in groups[c] if int(i) not in used)
        used.add(item); result.append(item)
    return np.array(result, dtype=np.int64)


class Meter:
    def __init__(self, calibration, cap):
        self.c = calibration; self.cap = float(cap); self.spent = 0.; self.counts = Counter(); self.events = []
        vals = list(self.c['reward_ms'].values()) + [self.c[k] for k in
               ('score_ms','setup_ms','uniform_ms','sid_ms','refine_ms')]
        if not np.isfinite(self.cap) or self.cap <= 0 or any(not np.isfinite(x) or x <= 0 for x in vals):
            raise ValueError('finite positive calibration/cap required')
    def fits(self, cost):
        return self.spent + cost <= self.cap
    def charge(self, kind, cost, n=1, samples=None):
        if not self.fits(cost):
            raise ValueError('operation was not affordable')
        self.spent += float(cost); self.counts[kind] += n
        self.events.append(dict(kind=kind, n=n, samples=samples,estimated_ms=float(cost), cumulative_ms=self.spent))
    def reward_cost(self, n):
        return self.c['reward_ms'][str(n)] if n else 0.


class Search:
    def __init__(self, model, calibration, cap, key, labels):
        self.model=model; self.meter=Meter(calibration, cap); self.rng=np.random.default_rng(key)
        self.labels=np.asarray(labels); self.cache={}; self.trace=[]; self.best=None; self.best_value=-np.inf
        self.stop='budget'; self.error=None
    def evaluate(self, state, candidates, kind):
        """Construction was prepaid; cache checks are included in construction estimate."""
        fresh=[]; seen=set()
        for action in candidates:
            key=tuple(map(int,action))
            if key in self.cache:
                self.meter.counts['cache_hits']+=1
            elif key in seen:
                self.meter.counts['duplicates']+=1
            else:
                seen.add(key); fresh.append(action)
        if fresh:
            self.meter.charge('reward_calls', self.meter.reward_cost(len(fresh)),samples=len(fresh))
            values=self.model.reward(np.repeat(state,len(fresh),axis=0), np.stack(fresh))
            if not np.isfinite(values).all():
                raise ValueError('non-finite model reward')
            self.meter.counts['reward_samples']+=len(fresh)
            self.meter.counts['fresh_evaluations']+=len(fresh)
            for action,value in zip(fresh,values):
                self.cache[tuple(map(int,action))]=float(value)
        result=[]
        for action in candidates:
            value=self.cache[tuple(map(int,action))]; result.append(value)
            self.meter.counts['emitted_candidates']+=1
            if value>self.best_value:
                self.best=action.copy(); self.best_value=value; self.trace.append(value)
        return result
    def base(self, state, count, grouped):
        coefficient=self.meter.c['sid_ms' if grouped else 'uniform_ms']
        size=min(count,self.meter.c['max_batch'])
        while size and not self.meter.fits(size*coefficient+self.meter.reward_cost(size)):
            size-=1
        if not size:return 0
        self.meter.charge('base_attempts',size*coefficient,size)
        actions=[generate(self.rng,self.model.n_items,self.model.slate_size,
                          self.groups if grouped else None,self.probabilities if grouped else None) for _ in range(size)]
        self.evaluate(state,actions,'base'); return size
    def refine(self, state, starts, grouped, reference=False):
        changes=0
        width=3 if reference else 8
        maximum=max((len(g) for g in self.groups),default=0) if grouped else len(self.order)
        for offset in range(0,3 if reference else maximum,width):
            for pos in range(self.model.slate_size):
                # Decide the slot count without constructing candidates. Setup already
                # prepaid these ranked group indices; each slot pays its legality scan.
                orders=[self.group_orders[int(self.labels[current[pos]])] if grouped else self.order for current in starts]
                slots=sum(max(0,min(width,len(order)-offset)) for order in orders)
                admitted=slots
                while admitted and not self.meter.fits(admitted*self.meter.c['refine_ms']+self.meter.reward_cost(admitted)):
                    admitted-=1
                if slots and not admitted:return changes,False
                if not admitted:continue
                self.meter.charge('refinement_attempts',admitted*self.meter.c['refine_ms'],admitted)
                actions=[];owners=[];attempted=0
                for w,current in enumerate(starts):
                    for item in orders[w][offset:offset+width]:
                        if attempted>=admitted:break
                        attempted+=1
                        if int(item)==int(current[pos]) or int(item) in current:
                            self.meter.counts['invalid_refinement_slots']+=1;continue
                        action=current.copy();action[pos]=int(item);actions.append(action);owners.append(w)
                    if attempted>=admitted:break
                self.meter.counts['legal_refinement_candidates']+=len(actions)
                values=self.evaluate(state,actions,'refine') if actions else []
                for action,value,w in zip(actions,values,owners):
                    if value>self.cache[tuple(map(int,starts[w]))]:
                        starts[w]=action.copy();changes+=1
                        self.meter.counts['accepted_refinements']+=1
                if admitted<slots:return changes,False
        return changes,True
    def plan(self,state,method,diagnostic=False,reference=False):
        minimum=self.meter.c['score_ms']+self.meter.c['setup_ms']+self.meter.reward_cost(1)
        if not self.meter.fits(minimum):raise ValueError('common legal anchor cannot fit cap')
        self.meter.charge('score_calls',self.meter.c['score_ms'],samples=1); self.meter.counts['score_samples']+=1
        scores=self.model.item_scores(state)[0]
        self.meter.charge('setup_calls',self.meter.c['setup_ms'])
        self.order,self.groups,self.probabilities=prepare(scores,self.labels)
        self.group_orders={int(c):g for c,g in zip(np.unique(self.labels),self.groups)}
        anchor=self.order[:self.model.slate_size].astype(np.int64)
        self.evaluate(state,[anchor],'anchor'); self.anchor=anchor.copy(); self.anchor_value=self.best_value
        grouped='sid' in method; refining=method=='flat_coordinate_refine' or 'hier' in method
        try:
            if diagnostic:
                starts=[anchor.copy()]
                while True:
                    changes,complete=self.refine(state,starts,grouped)
                    if not complete:break
                    if not changes:self.stop='converged';break
            elif reference:
                remaining=34
                while remaining:
                    count=self.base(state,remaining,True)
                    if not count:raise ValueError('reference cap too small')
                    remaining-=count
                ranked=sorted(self.cache,key=lambda a:-self.cache[a])[:4]
                self.refine(state,[np.array(a) for a in ranked],True,reference=True); self.stop='reference_complete'
            elif not refining:
                while self.base(state,32,grouped):pass
            else:
                while True:
                    if not self.base(state,32,grouped):break
                    ranked=sorted(self.cache,key=lambda a:-self.cache[a])[:4]
                    starts=[np.array(a) for a in ranked]
                    while True:
                        changes,complete=self.refine(state,starts,grouped)
                        if not complete:break
                        if not changes:break
                    if not complete:break
        except Exception as error:
            self.error=repr(error); self.stop='error_fallback'
        if self.best is None:raise RuntimeError('evaluated anchor missing')
        if len(set(self.best))!=self.model.slate_size or not np.all((self.best>=0)&(self.best<self.model.n_items)):
            raise ValueError('illegal selected slate')
        return self.best.copy(),dict(predicted_reward=self.best_value,anchor=self.anchor.tolist(),
            anchor_prediction=self.anchor_value,incumbent_trace=self.trace,cost=dict(self.meter.counts),
            events=self.meter.events,estimated_ms=self.meter.spent,cap_ms=self.meter.cap,
            unused_ms=max(0.,self.meter.cap-self.meter.spent),stop_reason=self.stop,error=self.error,
            base_cost_kind='sid_ms' if grouped else 'uniform_ms')


def validate_ledger(row, calibration):
    """Validate model batches, prepaid construction counts and exact declared cap."""
    c=calibration['coefficients'];cap=calibration['mid_cap_ms'];counts=Counter();total=0.
    if row['error'] or row['stop_reason']=='error_fallback':raise ValueError('unexpected decision error fallback')
    if row['cap_ms']!=cap:raise ValueError('decision cap differs from calibration')
    for event in row['events']:
        kind=event['kind'];n=event['n'];samples=event['samples']
        if not isinstance(n,int) or n<1:raise ValueError('invalid operation count')
        if kind=='reward_calls':
            if n!=1 or not isinstance(samples,int) or not 1<=samples<=c['max_batch']:raise ValueError('invalid reward invocation')
            expected=c['reward_ms'][str(samples)];counts['reward_samples']+=samples
        elif kind=='score_calls':
            if n!=1 or samples!=1:raise ValueError('invalid score invocation')
            expected=c['score_ms'];counts['score_samples']+=1
        elif kind=='setup_calls':
            if n!=1 or samples is not None:raise ValueError('invalid setup invocation')
            expected=c['setup_ms']
        elif kind=='base_attempts':
            # method-specific coefficient is bound by caller, not inferred from values.
            expected=n*c[row['base_cost_kind']]
        elif kind=='refinement_attempts':expected=n*c['refine_ms']
        else:raise ValueError('unknown ledger operation')
        if not np.isclose(event['estimated_ms'],expected,atol=1e-10,rtol=0):raise ValueError('event coefficient mismatch')
        counts[kind]+=n;total+=expected
        if not np.isclose(event['cumulative_ms'],total,atol=1e-8,rtol=0):raise ValueError('event accumulation mismatch')
    actual=row['cost']
    for k,v in counts.items():
        if actual.get(k,0)!=v:raise ValueError('operation count mismatch')
    for k in ('reward_calls','reward_samples','score_calls','score_samples','setup_calls','base_attempts','refinement_attempts'):
        if actual.get(k,0)!=counts.get(k,0):raise ValueError('extra operation count')
    if actual.get('fresh_evaluations',0)!=actual.get('reward_samples',0):raise ValueError('fresh evaluation mismatch')
    if actual.get('refinement_attempts',0)!=actual.get('invalid_refinement_slots',0)+actual.get('legal_refinement_candidates',0):raise ValueError('refinement slot mismatch')
    if actual.get('emitted_candidates',0)!=actual.get('base_attempts',0)+actual.get('legal_refinement_candidates',0)+1:raise ValueError('emission count mismatch')
    if actual.get('emitted_candidates',0)!=actual.get('fresh_evaluations',0)+actual.get('cache_hits',0)+actual.get('duplicates',0):raise ValueError('cache accounting mismatch')
    if total>cap or not np.isclose(total,row['estimated_ms'],atol=1e-8,rtol=0) or not np.isclose(row['unused_ms'],cap-total,atol=1e-8,rtol=0):raise ValueError('budget mismatch')
    return total



