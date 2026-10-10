"""R2-A additive planner: action-addressed random streams and explicit finite horizon."""
import hashlib
import itertools
import json
from contextlib import nullcontext
import numpy as np
from muslate.planning.search import BasePlanner
from muslate.planning.validity import validate_slates


def node_seed(decision_key, path):
    payload=json.dumps([int(decision_key),[list(map(int,a)) for a in path]],separators=(',',':'))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],'little')


class AddressedLookahead(BasePlanner):
    def __init__(self, model, proposal, depth=0, beam_width=3, gamma=.95):
        super().__init__(model,gamma)
        if depth<0 or beam_width<1:raise ValueError('invalid depth/beam')
        self.proposal=proposal;self.depth=depth;self.beam_width=beam_width
        self.last={}

    def pool(self,state,key,path):
        self.proposal.rng=np.random.default_rng(node_seed(key,path))
        return self.proposal.propose(state)[0]

    def _solve(self,state,h,key,path,root=None):
        actions=self.pool(state,key,path) if root is None else np.array(root,copy=True)
        validate_slates(actions,self.model.n_items,self.model.slate_size)
        rewards,next_states=self.model.step(np.repeat(state,len(actions),axis=0),actions)
        if h==0:
            idx=int(np.argmax(rewards));return float(rewards[idx]),[actions[idx].tolist()],actions,rewards
        top=np.argsort(-rewards,kind='stable')[:self.beam_width]
        best=-np.inf;sequence=None
        for idx in top:
            action=tuple(map(int,actions[idx]))
            child,seq,_,_=self._solve(next_states[idx:idx+1],h-1,key,path+(action,))
            value=float(rewards[idx]+self.gamma*child)
            if value>best:best=value;sequence=[list(action)]+seq
        return best,sequence,actions,rewards

    def plan(self,states,remaining_steps=None,budget=None,*,decision_key=0,root_candidates=None):
        if budget is not None:raise ValueError('R2-A is nominal-setting exploratory search, no hard-budget claim')
        state=np.atleast_2d(states)
        if len(state)!=1:raise ValueError('serial per-state decisions required')
        h=self._effective_depth(self.depth,remaining_steps)
        scope=self.model.decision() if hasattr(self.model,'decision') else nullcontext()
        with scope:
            value,seq,pool,rewards=self._solve(state,h,decision_key,(),root_candidates)
        self.last=dict(effective_depth=h,predicted_value=value,predicted_sequence=seq,
                       root_pool_sha256=hashlib.sha256(pool.astype('<i8').tobytes()).hexdigest(),
                       root_pool_size=len(pool),root_seed=node_seed(decision_key,()),
                       selected_immediate_reward=float(rewards[np.flatnonzero(np.all(pool==seq[0],axis=1))[0]]))
        return np.array(seq[:1],dtype=np.int64)


def exact_oracle(sim,state,horizon):
    """All ordered legal slates, all legal sequences; true state, evaluation only."""
    if horizon<1 or state.t+horizon>sim.cfg.horizon:raise ValueError('invalid exhaustive horizon')
    actions=np.array(list(itertools.permutations(range(sim.n_items),sim.slate_size)))
    def solve(s,h):
        best=-np.inf;chosen=None;seq=None
        for action in actions:
            reward,nxt=sim.step(s,action,noise=False)
            child,_,tail=solve(nxt,h-1) if h>1 else (0.,None,[])
            value=reward+sim.cfg.gamma*child
            if value>best:best=value;chosen=action.copy();seq=[action.tolist()]+tail
        return float(best),chosen,seq
    return solve(state,horizon)
