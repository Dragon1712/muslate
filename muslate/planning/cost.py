"""Measured operation weights and explicit estimated-compute budgets.

Units are calibrated milliseconds at a declared reference batch/hardware;
this linear estimate is not a wall-clock deadline or a FLOP measurement.
"""
from contextlib import contextmanager,nullcontext
from dataclasses import dataclass
from functools import wraps
import math,time


@dataclass(frozen=True)
class CostWeights:
    reward_ms: float
    transition_ms: float  # joint reward + transition call, disjoint from above
    score_ms: float
    proposal_ms: float

    def __post_init__(self):
        if any(not math.isfinite(x) or x<=0 for x in vars(self).values()):
            raise ValueError('positive finite calibrated weights required')


@dataclass
class PlanningCost:
    reward_only_calls: int=0
    transition_calls: int=0
    score_queries: int=0
    proposal_candidates: int=0
    unique_slates: int=0
    emitted_candidates: int=0
    proposal_time_ms: float=0.  # exclusive of timed inner model work
    proposal_wall_ms: float=0.
    planning_time_ms: float=0.
    model_time_ms: float=0.

    def estimate(self,w:CostWeights):
        return (w.reward_ms*self.reward_only_calls+w.transition_ms*self.transition_calls
                +w.score_ms*self.score_queries+w.proposal_ms*self.proposal_candidates)


class BudgetExceeded(RuntimeError):pass


@dataclass
class PlanningBudget:
    max_compute: float
    weights: CostWeights
    used: float=0.

    def __post_init__(self):
        if not math.isfinite(self.max_compute) or self.max_compute<0 or self.used!=0:
            raise ValueError('new nonnegative finite budget required')

    def spend(self,cost):
        if self.used+cost>self.max_compute+1e-12:
            raise BudgetExceeded('calibrated compute budget exhausted before operation')
        self.used+=cost


def planning_decision(fn):
    @wraps(fn)
    def wrapped(self,states,remaining_steps=None,budget=None):
        if budget is not None:
            if not isinstance(budget,PlanningBudget) or not hasattr(self.model,'decision'):
                raise ValueError('hard budget requires PlanningBudget and CountingModel')
            if self.adjust is not None:
                raise ValueError('hard budget cannot accept an unmetered external adjust hook')
            proposal=getattr(self,'proposal',None)
            if proposal is not None:
                from muslate.planning.proposal import ProposalPolicy
                from muslate.planning.abstractions import SIDProposal
                if type(proposal) not in (ProposalPolicy,SIDProposal) or getattr(proposal,'model',None) is not self.model:
                    raise ValueError('hard budget requires built-in proposal bound to the same CountingModel')
        scope=self.model.decision(budget) if hasattr(self.model,'decision') else nullcontext()
        with scope:return fn(self,states,remaining_steps=remaining_steps,budget=budget)
    return wrapped


def timed_proposal(fn):
    @wraps(fn)
    def wrapped(self,*args,**kwargs):
        scope=self.model.proposal_scope() if hasattr(self.model,'proposal_scope') else nullcontext()
        with scope:
            result=fn(self,*args,**kwargs)
            if hasattr(self.model,'record_emitted'):self.model.record_emitted(result)
            return result
    return wrapped
