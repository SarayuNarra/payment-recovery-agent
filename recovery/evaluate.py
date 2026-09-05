"""
Baselines and metrics.

The baselines are the point of this file. A recovery agent that reports
"recovered 34%" in isolation has proved nothing -- the question is always
"compared to what?". These four are what production systems actually do today.
"""

from __future__ import annotations

from datetime import timedelta
from typing import List

from .agent import Gatekeeper, TxnResult, AuditEntry, next_business_hour
from .corpus import Transaction
from .simulator import (
    CONTACT_ACTIONS,
    RETRY_ALT_RAIL,
    RETRY_SAME_RAIL,
    SEND_PAYMENT_LINK,
    Simulator,
)


class FixedStrategy:
    """Runs the same hard-coded ladder for every transaction."""

    def __init__(self, name: str, ladder: List[tuple], sim: Simulator,
                 respect_gates: bool = True):
        self.name = name
        self.ladder = ladder
        self.sim = sim
        self.respect_gates = respect_gates

    def run_batch(self, txns: List[Transaction]):
        return [self._run_one(t) for t in txns], None

    def _run_one(self, txn: Transaction) -> TxnResult:
        gate = Gatekeeper()
        recovered = cost = 0
        outcome = "UNRECOVERED"
        audit: List[AuditEntry] = []
        attempt_no = 0

        for step, (action, offset_h) in enumerate(self.ladder):
            at = txn.failed_at + timedelta(hours=offset_h)
            if action in CONTACT_ACTIONS:
                at = next_business_hour(at)
            if self.respect_gates:
                blocked = gate.check(txn, action, at)
                if blocked:
                    audit.append(AuditEntry(
                        txn.txn_id, at.isoformat(), step, "n/a", 0.0,
                        action, "BLOCKED", blocked))
                    continue
            gate.record(action, at)
            attempt_no += 1
            res = self.sim.attempt(txn, action, at, attempt_no)
            recovered += res.recovered_paise
            cost += res.cost_paise
            audit.append(AuditEntry(
                txn.txn_id, at.isoformat(), step, "n/a", 0.0, action,
                "EXECUTED", self.name, None, res.outcome,
                round(res.recovered_paise / 100, 2),
                round(res.cost_paise / 100, 2)))
            if res.outcome == "SUCCESS":
                outcome = "RECOVERED"
                break
            if res.outcome == "DOUBLE_CHARGE":
                outcome = "DOUBLE_CHARGE"
                break

        return TxnResult(txn.txn_id, recovered, cost, gate.actions, outcome, audit)


def build_baselines(sim: Simulator) -> dict:
    return {
        "do_nothing": FixedStrategy("do_nothing", [], sim),
        "immediate_retry_x2": FixedStrategy(
            "immediate_retry_x2",
            [(RETRY_SAME_RAIL, 0.02), (RETRY_SAME_RAIL, 0.1)], sim),
        "fixed_24h_x3": FixedStrategy(
            "fixed_24h_x3",
            [(RETRY_SAME_RAIL, 24), (RETRY_SAME_RAIL, 48), (RETRY_SAME_RAIL, 72)], sim),
        "link_blast": FixedStrategy(
            "link_blast",
            [(SEND_PAYMENT_LINK, 0.1), (SEND_PAYMENT_LINK, 26),
             (RETRY_ALT_RAIL, 50)], sim),
    }


def score(txns: List[Transaction], results: List[TxnResult], sim: Simulator) -> dict:
    at_risk = sum(t.amount_paise for t in txns)
    recovered = sum(r.recovered_paise for r in results)
    cost = sum(r.cost_paise for r in results)
    actions = sum(r.actions_taken for r in results)
    n_rec = sum(1 for r in results if r.outcome == "RECOVERED")
    doubles = sum(1 for r in results if r.outcome == "DOUBLE_CHARGE")

    # Wasted charge attempts: a debit fired at an instrument that could never
    # have worked at that moment. Scored from hidden state, for reporting only.
    wasted = 0
    for t, r in zip(txns, results):
        for e in r.audit:
            if e.decision == "EXECUTED" and e.outcome == "FAIL":
                if t.hidden.risk_blocked or (
                    t.hidden.instrument_dead
                    and e.proposed_action == RETRY_SAME_RAIL
                ):
                    wasted += 1

    ceiling = sum(t.amount_paise for t in txns if sim.recoverable_ceiling(t))

    return {
        "at_risk_inr": round(at_risk / 100, 2),
        "recoverable_ceiling_inr": round(ceiling / 100, 2),
        "recovered_inr": round(recovered / 100, 2),
        "recovery_rate_value": round(recovered / at_risk, 4) if at_risk else 0.0,
        "recovery_rate_count": round(n_rec / len(txns), 4),
        "share_of_ceiling": round(recovered / ceiling, 4) if ceiling else 0.0,
        "cost_inr": round(cost / 100, 2),
        "net_inr": round((recovered - cost) / 100, 2),
        "actions": actions,
        "actions_per_txn": round(actions / len(txns), 2),
        "cost_per_rupee_recovered": (
            round(cost / recovered, 4) if recovered else None
        ),
        "double_charges": doubles,
        "wasted_attempts_on_dead_instruments": wasted,
    }
