"""
Learned intervention timing.

Hand-writing "retry at +2h, then +24h" is guesswork. A real merchant has years
of retry logs, so the agent should estimate the timing curve instead of assuming
it.

We simulate having those logs: on a *train split* of transactions, we fire
randomised probes (random action, random delay) and record whether each one
worked. From that we estimate

    P(success | diagnosis class, flow, action, delay bucket)

with Laplace smoothing and backoff for sparse cells. The agent then ranks
(action, delay) pairs by expected value and walks the top three.

The test split is never probed, so every number reported in run_eval.py is
out-of-sample. The probes only ever condition on observable fields -- the
diagnosis comes from the same diagnoser the agent uses at test time.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Optional

from .corpus import Transaction
from .simulator import (
    RETRY_ALT_RAIL,
    RETRY_SAME_RAIL,
    SEND_PAYMENT_LINK,
    COST_LINK,
    COST_RETRY,
    Simulator,
)

# Delay buckets in hours. Coarse on purpose -- fine buckets just add variance.
BUCKETS = [0.25, 2.0, 8.0, 24.0, 72.0, 168.0, 312.0]
ACTIONS = [RETRY_SAME_RAIL, RETRY_ALT_RAIL, SEND_PAYMENT_LINK]

PRIOR_STRENGTH = 6.0
PRIOR_P = 0.25


class TimingModel:
    def __init__(self):
        # counts[key] = [successes, trials]
        self.full = defaultdict(lambda: [0.0, 0.0])
        self.no_flow = defaultdict(lambda: [0.0, 0.0])
        self.marginal = defaultdict(lambda: [0.0, 0.0])
        self.n_probes = 0

    def observe(self, dclass, flow, action, bucket, success):
        self.n_probes += 1
        for store, key in (
            (self.full, (dclass, flow, action, bucket)),
            (self.no_flow, (dclass, action, bucket)),
            (self.marginal, (action, bucket)),
        ):
            store[key][0] += float(success)
            store[key][1] += 1.0

    def p(self, dclass, flow, action, bucket) -> float:
        """Smoothed estimate with backoff from specific to marginal cells."""
        s, n = self.marginal[(action, bucket)]
        base = (s + PRIOR_STRENGTH * PRIOR_P) / (n + PRIOR_STRENGTH)

        s2, n2 = self.no_flow[(dclass, action, bucket)]
        mid = (s2 + PRIOR_STRENGTH * base) / (n2 + PRIOR_STRENGTH)

        s3, n3 = self.full[(dclass, flow, action, bucket)]
        return (s3 + PRIOR_STRENGTH * mid) / (n3 + PRIOR_STRENGTH)

    def ranked_ladder(self, dclass: str, flow: str, amount_paise: int,
                      k: int = 3) -> List[tuple]:
        """Top-k (action, delay_hours) by expected value, one per action type."""
        scored = []
        for action in ACTIONS:
            cost = COST_RETRY if action != SEND_PAYMENT_LINK else COST_LINK
            for b in BUCKETS:
                ev = self.p(dclass, flow, action, b) * amount_paise - cost
                scored.append((ev, action, b))
        scored.sort(reverse=True)

        ladder, used_actions = [], defaultdict(int)
        for ev, action, b in scored:
            if ev <= 0:
                continue
            if used_actions[action] >= 2:      # don't stack four identical retries
                continue
            if any(abs(b - ob) < 1e-9 and action == oa for oa, ob in ladder):
                continue
            ladder.append((action, b))
            used_actions[action] += 1
            if len(ladder) >= k:
                break
        # Later steps must happen later in wall-clock time.
        ladder.sort(key=lambda x: x[1])
        return ladder


def fit(train: List[Transaction], diagnoser, sim: Simulator,
        probes_per_txn: int = 4, seed: int = 3) -> TimingModel:
    rng = random.Random(seed)
    model = TimingModel()
    diags = diagnoser.diagnose_batch(train)

    for txn, d in zip(train, diags):
        for i in range(probes_per_txn):
            action = rng.choice(ACTIONS)
            bucket = rng.choice(BUCKETS)
            jitter = 1.0 + rng.uniform(-0.15, 0.15)
            at = txn.failed_at + _hours(bucket * jitter)
            res = sim.attempt(txn, action, at, attempt_no=100 + i)
            model.observe(
                d.dclass, txn.flow, action, bucket,
                res.outcome == "SUCCESS",
            )
    return model


def _hours(h: float):
    from datetime import timedelta
    return timedelta(hours=h)


def report(model: TimingModel, dclass: str, flow: str = "checkout") -> str:
    lines = [f"{dclass} / {flow}"]
    for action in ACTIONS:
        cells = " ".join(
            f"{b:>5.0f}h:{model.p(dclass, flow, action, b)*100:5.1f}%"
            for b in BUCKETS
        )
        lines.append(f"  {action:<18} {cells}")
    return "\n".join(lines)
