"""
The recovery agent.

Structure is deliberately three separate layers, because "every money action
explainable, bounded and gated" is easier to demonstrate when the gates are not
tangled into the policy:

  1. Diagnosis   (diagnoser.py) - why did this fail?
  2. Policy      (plan_actions)  - what sequence would recover it?
  3. Gates       (Gatekeeper)    - is each action allowed and worth doing?

Every action, allowed or blocked, is written to the audit trail with the reason.
An action that a gate rejects is never sent to the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

from .corpus import Transaction
from .diagnoser import (
    Diagnosis,
    FUNDS_TIMED,
    INSTRUMENT_DEAD,
    LIMIT_TIMED,
    RISK_TERMINAL,
    TRANSIENT_SHORT,
    UNKNOWN,
    USER_INTENT_DROP,
)
from .simulator import (
    CHARGING_ACTIONS,
    CONTACT_ACTIONS,
    COST_LINK,
    COST_RETRY,
    GIVE_UP,
    RETRY_ALT_RAIL,
    RETRY_SAME_RAIL,
    SEND_PAYMENT_LINK,
    AttemptResult,
    Simulator,
)

# --- Bounds. These are policy, not tuning knobs, and are enforced centrally. ---
MAX_ACTIONS_PER_TXN = 4
MAX_CHARGE_ATTEMPTS = 3
MAX_CONTACTS_PER_72H = 2
QUIET_HOURS = (21, 8)          # no customer contact 21:00-08:00 local
MIN_GAP_BETWEEN_CONTACTS_H = 24
RECOVERY_WINDOW_DAYS = 16      # stop chasing after this
MIN_EV_PAISE = 0               # never take an action with negative expected value
MIN_GAP_BETWEEN_ACTIONS_MIN = 15   # two debits must never leave at the same instant


@dataclass
class AuditEntry:
    txn_id: str
    at: str
    step: int
    diagnosis: str
    confidence: float
    proposed_action: str
    decision: str            # EXECUTED | BLOCKED
    reason: str
    expected_value_inr: Optional[float] = None
    outcome: Optional[str] = None
    recovered_inr: float = 0.0
    cost_inr: float = 0.0


@dataclass
class TxnResult:
    txn_id: str
    recovered_paise: int
    cost_paise: int
    actions_taken: int
    outcome: str
    audit: List[AuditEntry] = field(default_factory=list)


class Gatekeeper:
    """Hard bounds. Nothing reaches the simulator without passing every check."""

    def __init__(self):
        self.contacts: List[datetime] = []
        self.charges = 0
        self.actions = 0

    def check(
        self, txn: Transaction, action: str, at: datetime
    ) -> Optional[str]:
        if self.actions >= MAX_ACTIONS_PER_TXN:
            return f"action budget exhausted ({MAX_ACTIONS_PER_TXN})"
        if at - txn.failed_at > timedelta(days=RECOVERY_WINDOW_DAYS):
            return f"outside {RECOVERY_WINDOW_DAYS}d recovery window"
        if action in CHARGING_ACTIONS and self.charges >= MAX_CHARGE_ATTEMPTS:
            return f"charge attempt cap reached ({MAX_CHARGE_ATTEMPTS})"
        if action in CONTACT_ACTIONS:
            hi, lo = QUIET_HOURS
            if at.hour >= hi or at.hour < lo:
                return f"quiet hours ({at.hour:02d}:00)"
            recent = [c for c in self.contacts if at - c < timedelta(hours=72)]
            if len(recent) >= MAX_CONTACTS_PER_72H:
                return f"contact fatigue cap ({MAX_CONTACTS_PER_72H} per 72h)"
            if self.contacts and at - self.contacts[-1] < timedelta(
                hours=MIN_GAP_BETWEEN_CONTACTS_H
            ):
                return "min gap between contacts not met"
        return None

    def record(self, action: str, at: datetime) -> None:
        self.actions += 1
        if action in CHARGING_ACTIONS:
            self.charges += 1
        if action in CONTACT_ACTIONS:
            self.contacts.append(at)


def next_business_hour(at: datetime) -> datetime:
    """Push a contact out of quiet hours instead of dropping it."""
    hi, lo = QUIET_HOURS
    if at.hour >= hi:
        return (at + timedelta(days=1)).replace(hour=lo + 1, minute=0, second=0)
    if at.hour < lo:
        return at.replace(hour=lo + 1, minute=0, second=0)
    return at


def plan_actions(
    txn: Transaction, d: Diagnosis
) -> List[tuple]:
    """
    Returns [(action, offset_hours)] -- the intervention ladder for this
    diagnosis. Ordering matters: cheap silent debits first where they can work,
    customer contact only when the customer is actually needed.
    """
    t0 = 0.0

    if d.dclass == RISK_TERMINAL:
        return [(GIVE_UP, 0.0)]

    if d.dclass == INSTRUMENT_DEAD:
        # Retrying the same instrument is guaranteed waste. Go straight to a
        # different rail, then ask the customer for a new instrument.
        return [(RETRY_ALT_RAIL, 0.5), (SEND_PAYMENT_LINK, 2.0), (SEND_PAYMENT_LINK, 48.0)]

    if d.dclass == TRANSIENT_SHORT:
        w = max(0.25, min(d.suggested_wait_h, 6.0))
        return [(RETRY_SAME_RAIL, w), (RETRY_SAME_RAIL, w + 3), (RETRY_ALT_RAIL, w + 20)]

    if d.dclass == FUNDS_TIMED:
        # Money arrives on payday. Guess conservatively from the wait hint, and
        # keep one late attempt for the following cycle.
        w = max(12.0, min(d.suggested_wait_h, 96.0))
        if txn.flow == "checkout":
            # Intent decays fast; a link now is worth more than a debit later.
            return [(SEND_PAYMENT_LINK, 1.0), (RETRY_SAME_RAIL, w), (RETRY_SAME_RAIL, w + 48)]
        return [(RETRY_SAME_RAIL, w), (RETRY_SAME_RAIL, w + 48), (SEND_PAYMENT_LINK, w + 72)]

    if d.dclass == LIMIT_TIMED:
        midnight = (txn.failed_at + timedelta(days=1)).replace(hour=0, minute=30)
        h = (midnight - txn.failed_at).total_seconds() / 3600.0
        return [(RETRY_ALT_RAIL, 0.5), (RETRY_SAME_RAIL, h), (RETRY_SAME_RAIL, h + 24)]

    if d.dclass == USER_INTENT_DROP:
        # Nothing is broken, so a silent retry cannot help: authentication needs
        # the customer. Contact fast, because intent halves in hours.
        return [(SEND_PAYMENT_LINK, 0.25), (SEND_PAYMENT_LINK, 26.0), (RETRY_SAME_RAIL, 50.0)]

    # UNKNOWN: one cheap probe, then re-engage the customer.
    return [(RETRY_SAME_RAIL, 2.0), (SEND_PAYMENT_LINK, 6.0), (RETRY_ALT_RAIL, 30.0)]


def expected_value(txn: Transaction, action: str, d: Diagnosis, step: int) -> int:
    """
    Crude EV in paise. It only needs to be good enough to veto obviously bad
    actions -- chasing a 49-rupee failure with three retries, for instance.
    Deliberately does not use hidden state.
    """
    if action == GIVE_UP:
        return 0
    prior = {
        TRANSIENT_SHORT: 0.55,
        FUNDS_TIMED: 0.42,
        LIMIT_TIMED: 0.5,
        INSTRUMENT_DEAD: 0.3,
        USER_INTENT_DROP: 0.28,
        UNKNOWN: 0.22,
        RISK_TERMINAL: 0.0,
    }.get(d.dclass, 0.2)
    p = prior * (0.62 ** step) * (0.6 + 0.4 * d.confidence)
    if action == SEND_PAYMENT_LINK:
        p *= 0.8
    cost = COST_RETRY if action in CHARGING_ACTIONS else COST_LINK
    return int(p * txn.amount_paise) - cost


class RecoveryAgent:
    """Diagnose once, then walk a bounded intervention ladder."""

    def __init__(self, diagnoser, simulator: Simulator, name: str = "agent",
                 timing_model=None):
        self.diagnoser = diagnoser
        self.sim = simulator
        self.name = name
        # If present, the learned timing curve replaces the hand-written ladder
        # for every diagnosis except the terminal one, which stays hard-coded:
        # "never retry a risk block" is a policy decision, not a statistical one.
        self.timing_model = timing_model

    def run_batch(self, txns: List[Transaction]) -> tuple:
        diags = self.diagnoser.diagnose_batch(txns)
        results = [self._run_one(t, d) for t, d in zip(txns, diags)]
        return results, diags

    def _run_one(self, txn: Transaction, d: Diagnosis) -> TxnResult:
        gate = Gatekeeper()
        audit: List[AuditEntry] = []
        recovered = 0
        cost = 0
        outcome = "UNRECOVERED"
        attempt_no = 0
        last_at: Optional[datetime] = None

        if self.timing_model is not None and d.dclass != RISK_TERMINAL:
            ladder = self.timing_model.ranked_ladder(
                d.dclass, txn.flow, txn.amount_paise)
            if not ladder:
                ladder = [(GIVE_UP, 0.0)]
        else:
            ladder = plan_actions(txn, d)

        for step, (action, offset_h) in enumerate(ladder):
            at = txn.failed_at + timedelta(hours=offset_h)
            if action in CONTACT_ACTIONS:
                at = next_business_hour(at)
            # The learned ladder can rank two actions into the same delay
            # bucket. Firing both at the same instant is not something a
            # payment system can actually do, so space them.
            if last_at is not None and at <= last_at + timedelta(
                    minutes=MIN_GAP_BETWEEN_ACTIONS_MIN):
                at = last_at + timedelta(minutes=MIN_GAP_BETWEEN_ACTIONS_MIN)
                if action in CONTACT_ACTIONS:
                    at = next_business_hour(at)

            if action == GIVE_UP:
                audit.append(AuditEntry(
                    txn.txn_id, at.isoformat(), step, d.dclass, d.confidence,
                    GIVE_UP, "EXECUTED",
                    f"terminal diagnosis; no action taken ({d.rationale})",
                ))
                outcome = "ABANDONED_TERMINAL"
                break

            blocked = gate.check(txn, action, at)
            if blocked:
                audit.append(AuditEntry(
                    txn.txn_id, at.isoformat(), step, d.dclass, d.confidence,
                    action, "BLOCKED", blocked,
                ))
                continue

            ev = expected_value(txn, action, d, step)
            if ev <= MIN_EV_PAISE:
                audit.append(AuditEntry(
                    txn.txn_id, at.isoformat(), step, d.dclass, d.confidence,
                    action, "BLOCKED",
                    "negative expected value", round(ev / 100, 2),
                ))
                continue

            gate.record(action, at)
            last_at = at
            attempt_no += 1
            res: AttemptResult = self.sim.attempt(txn, action, at, attempt_no)
            cost += res.cost_paise
            recovered += res.recovered_paise

            audit.append(AuditEntry(
                txn.txn_id, at.isoformat(), step, d.dclass, d.confidence,
                action, "EXECUTED", d.rationale, round(ev / 100, 2),
                res.outcome, round(res.recovered_paise / 100, 2),
                round(res.cost_paise / 100, 2),
            ))

            if res.outcome == "SUCCESS":
                outcome = "RECOVERED"
                break
            if res.outcome == "DOUBLE_CHARGE":
                outcome = "DOUBLE_CHARGE"
                break

        return TxnResult(
            txn.txn_id, recovered, cost, gate.actions, outcome, audit
        )