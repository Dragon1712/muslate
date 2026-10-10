"""Slate-conditioned latent user world model.

    (h_t, slate_t) -> (r_t, h_{t+1})

The model is deliberately small -- it is not the contribution, it exists so that
planning can run against learned dynamics.  Three design choices matter:

1. **Order-aware slate encoding** for the *dynamics*.  A slate is an ordered
   set, so the encoder uses a positional discount plus a learned position
   embedding rather than a plain mean over item embeddings.

2. **Item-decomposed reward.**  The reward head is

       r(h, A) = sum_k  w_k * [ <g(h), e_{i_k}> + q(h)[i_k] ]

   i.e. an additive, item-level score with a per-item bias.  This matters for
   two reasons.  First it matches how slate rewards are actually generated, so
   it generalises to slates the behaviour policy never showed -- a monolithic
   ``MLP(h, slate_emb)`` reward head does not, and when it is paired with a
   planner it gets *exploited* (the planner finds the region where the head
   over-estimates).  Second, it shares its parameters with ``item_scores``, so
   the proposal policy and the value estimate can never disagree about which
   items are good.  The per-item bias ``q(h)[i]`` is what lets the model express
   delayed saturation: a category that has been over-shown is worth less
   regardless of the user's interest direction.

3. **Multi-step training.**  The dynamics head is trained to predict rewards
   several steps ahead through its own rollouts, otherwise it is useless in the
   H > 1 regime we care about.

The model is instantiated as an ensemble so disagreement-based epistemic
uncertainty is available for conservative planning.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from muslate.config import WorldModelConfig


class SlateEncoder(nn.Module):
    """Order-aware slate embedding used by the dynamics head."""

    def __init__(self, n_items: int, slate_size: int, emb_dim: int, pos_discount: float = 0.8):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.pos_emb = nn.Embedding(slate_size, emb_dim)
        self.out = nn.Linear(emb_dim, emb_dim)
        self.register_buffer(
            "pos_w",
            torch.tensor([pos_discount ** k for k in range(slate_size)], dtype=torch.float32),
        )
        nn.init.normal_(self.item_emb.weight, std=0.1)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, slates: torch.Tensor) -> torch.Tensor:
        """slates [B, K] long -> slate embedding [B, emb_dim]."""
        e = self.item_emb(slates)                       # [B, K, de]
        e = e * self.pos_w[None, :, None]
        e = e + self.pos_emb.weight[None, :, :]
        return torch.tanh(self.out(e.sum(dim=1)))


class WorldModel(nn.Module):
    def __init__(
        self,
        cfg: WorldModelConfig,
        n_items: int,
        slate_size: int,
        true_state_dim: int | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_items = n_items
        self.slate_size = slate_size
        self.true_state_dim = true_state_dim

        de, dh = cfg.slate_emb_dim, cfg.latent_dim
        self.slate_enc = SlateEncoder(n_items, slate_size, de)

        # history encoder: consumes (slate embedding, observed reward)
        self.gru = nn.GRUCell(de + 1, dh)
        self.h0 = nn.Parameter(torch.zeros(dh))

        # item-decomposed reward:  r = sum_k w_k ( <g(h), e_k> + q(h)[k] )
        self.g_head = nn.Sequential(
            nn.Linear(dh, cfg.hidden_dim), nn.ReLU(), nn.Linear(cfg.hidden_dim, de)
        )
        self.q_head = nn.Sequential(
            nn.Linear(dh, cfg.hidden_dim), nn.ReLU(), nn.Linear(cfg.hidden_dim, n_items)
        )

        # -- intra-slate interaction term -------------------------------------
        # The unary head above scores each item independently, so it has no way
        # to express the environment's diversity penalty, which is a sum over
        # item PAIRS.  W is kept symmetric and initialised at identity because
        # the simulator's term is exactly <z_j, z_k>.
        self.pairwise = bool(cfg.pairwise)
        if self.pairwise:
            self.pair_proj = nn.Linear(de, de, bias=False)
            with torch.no_grad():
                self.pair_proj.weight.copy_(torch.eye(de))
            pair_w = self.slate_enc.pos_w[:, None] * self.slate_enc.pos_w[None, :]
            self.register_buffer(
                "pair_mask", pair_w * torch.triu(torch.ones_like(pair_w), diagonal=1)
            )

        # latent dynamics
        self.dyn = nn.Sequential(
            nn.Linear(dh + de, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, dh),
        )

        self.state_head = nn.Linear(dh, true_state_dim) if cfg.oracle_state else None
        self.register_buffer("all_items", torch.arange(n_items, dtype=torch.long))

    # -- reward / item scores ---------------------------------------------
    def item_value(self, h: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """f(h, e_i) for explicit item ids.

        Accepts any leading batch shape: (..., dh) with (..., L) -> (..., L).
        """
        shape = h.shape[:-1]
        hf = h.reshape(-1, h.shape[-1])
        ids = item_ids.reshape(-1, item_ids.shape[-1])

        e = self.slate_enc.item_emb(ids)                          # [B, L, de]
        g = self.g_head(hf)                                       # [B, de]
        q = self.q_head(hf)                                       # [B, N]
        inner = (g[:, None, :] * e).sum(dim=-1)                   # [B, L]
        out = inner + q.gather(1, ids)
        return out.reshape(*shape, ids.shape[-1])

    def item_scores(self, h: torch.Tensor) -> torch.Tensor:
        """Myopic score of every item.  (..., dh) -> (..., N).

        Computed as a single matmul, and by construction consistent with
        ``predict_reward``.
        """
        shape = h.shape[:-1]
        hf = h.reshape(-1, h.shape[-1])
        g = self.g_head(hf)                                       # [B, de]
        q = self.q_head(hf)                                       # [B, N]
        scores = g @ self.slate_enc.item_emb.weight.t() + q
        return scores.reshape(*shape, self.n_items)

    def predict_unary(self, h: torch.Tensor, slates: torch.Tensor) -> torch.Tensor:
        """Unary part of the reward -- the part shared with ``item_scores``.

        Kept as a separate entry point so the "reward is decomposable into item
        scores" property can still be asserted; after adding the pairwise term
        the *whole* reward is no longer decomposable, only this part is.
        """
        shape = h.shape[:-1]
        hf = h.reshape(-1, h.shape[-1])
        sf = slates.reshape(-1, slates.shape[-1])
        v = self.item_value(hf, sf)                               # [B, K]
        r = (v * self.slate_enc.pos_w[None, :]).sum(dim=-1)
        return r.reshape(shape)

    def pair_value(self, h: torch.Tensor, slates: torch.Tensor) -> torch.Tensor:
        """Intra-slate interaction: ``sum_{j<k} w_j w_k * e_j^T W e_k``.

        Why this term has to exist: the simulator's diversity penalty is a
        function of item *pairs*, while a purely unary head can only express
        per-item contributions.  Without it the world model is **structurally**
        unable to represent the very term that makes the slate objective
        combinatorial -- and a structure-seeking search (the SID hierarchy) is
        then systematically misled under learned dynamics.

        ``W`` is enforced symmetric; the mask carries the same positional
        weighting (``w_j * w_k``) the environment uses.
        """
        shape = slates.shape[:-1]
        sf = slates.reshape(-1, slates.shape[-1])
        e = self.slate_enc.item_emb(sf)                           # [B, K, de]
        w = self.pair_proj.weight
        w = 0.5 * (w + w.t())                                     # enforce symmetry
        ew = e @ w                                                # [B, K, de]
        gram = ew @ e.transpose(-1, -2)                           # [B, K, K] = e_j^T W e_k
        out = (gram * self.pair_mask).sum(dim=(1, 2))
        return out.reshape(shape)

    def predict_reward(self, h: torch.Tensor, slates: torch.Tensor) -> torch.Tensor:
        """(..., dh) with (..., K) -> (...) reward: unary (+ pairwise)."""
        r = self.predict_unary(h, slates)
        if self.pairwise:
            r = r + self.pair_value(h, slates)
        return r

    # -- dynamics -----------------------------------------------------------
    def step_latent(self, h: torch.Tensor, slates: torch.Tensor) -> torch.Tensor:
        """(..., dh) with (..., K) -> (..., dh)."""
        shape = h.shape[:-1]
        hf = h.reshape(-1, h.shape[-1])
        sf = slates.reshape(-1, slates.shape[-1])
        se = self.slate_enc(sf)                                   # [B, de]
        nxt = hf + self.dyn(torch.cat([hf, se], dim=-1))
        return nxt.reshape(*shape, h.shape[-1])

    # -- sequence encoding --------------------------------------------------
    def rollout(
        self, slates: torch.Tensor, rewards: torch.Tensor, return_final: bool = False
    ):
        """Teacher-forced posterior states.

        slates [B, T, K], rewards [B, T] -> states [B, T, dh], where ``states[:, t]``
        is the belief *before* slate t, i.e. the belief used to predict r_t.
        With ``return_final`` also returns the belief after the whole history.
        """
        b, t, _k = slates.shape
        h = self.h0[None].expand(b, -1).contiguous()
        out = []
        for step in range(t):
            se = self.slate_enc(slates[:, step])
            out.append(h)
            h = self.gru(torch.cat([se, rewards[:, step : step + 1]], dim=-1), h)
        states = torch.stack(out, dim=1) if t > 0 else h.new_zeros(b, 0, h.shape[-1])
        if return_final:
            return states, h
        return states

    def encode_history(self, slates: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Latent belief after the whole history: slates [B, T, K], rewards [B, T]."""
        b = slates.shape[0]
        if slates.shape[1] == 0:
            return self.h0[None].expand(b, -1).contiguous()
        _states, h = self.rollout(slates, rewards, return_final=True)
        return h

    def forward(self, slates: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Posterior states [B, T, dh]."""
        return self.rollout(slates, rewards)
