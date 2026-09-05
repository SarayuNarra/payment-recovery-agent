"""
Outcome model.

The simulator is the only component that can read HiddenState. It takes an
action proposed by a strategy at a given instant and decides what actually
happened. It is deterministic given a seed, so every strategy is evaluated
against exactly the same world.

Costs are in paise so the economics stay in integers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .corpus import Transaction

# Actions a strategy may take.
RETRY_SAME_RAIL = "RETRY_SAME_RAIL"
RETRY_ALT_RAIL = "RETRY_ALT_RAIL"
SEND_PAYMENT_LINK = "SEND_PAYMENT_LINK"
GIVE_UP = "GIVE_UP"

CHARGING_ACTIONS = {RETRY_SAME_RAIL, RETRY_ALT_RAIL}
CONTACT_ACTIONS = {SEND_PAYMENT_LINK}

# Unit economics (paise).
COST_RETRY = 200          # gateway attempt + ops overhead
COST_LINK = 35            # SMS/WhatsApp delivery
COST_DOUBLE_CHARGE = 15000  # refund handling + support ticket + goodwill

BASE_P_SAME_RAIL = 0.93
BASE_P_ALT_RAIL = 0.88
BASE_P_LINK = 0.62  # customer must act, so a link is never as good as a debit


@dataclass
class AttemptResult:
    action: str
    at: datetime
    success: bool
    recovered_paise: int
    cost_paise: int
    outcome: str          # SUCCESS | FAIL | DOUBLE_CHARGE | SKIPPED
    failure_code: Optional[str] = None


def intent_factor(txn: Transaction, at: datetime) -> float:
    """How likely the customer still wants to complete, as time passes."""
    hours = max(0.0, (at - txn.failed_at).total_seconds() / 3600.0)
    hl = txn.hidden.intent_half_life_h
    if hl > 1e5:
        return 1.0
    return 0.5 ** (hours / hl)


def _blocked_reason(txn: Transaction, action: str, at: datetime) -> Optional[str]:
    """Hard blockers. Returns the failure code, or None if the path is clear."""
    h = txn.hidden

    if h.risk_blocked:
        # Risk holds follow the customer, not the instrument. Nothing helps.
        return "RISK_DECLINED_RZP"

    if h.funds_available_at and at < h.funds_available_at:
        # A link cannot conjure money either.
        return "INSUFFICIENT_FUNDS"

    if h.outage_end_at and at < h.outage_end_at:
        if action == RETRY_SAME_RAIL:
            return "ISSUER_DOWN"
        # An alternate rail or a link routes around a single issuer outage.

    if h.instrument_dead and action == RETRY_SAME_RAIL:
        return txn.error_code

    if h.limit_reset_at and at < h.limit_reset_at and action == RETRY_SAME_RAIL:
        return "LIMIT_EXCEEDED"

    if action == RETRY_ALT_RAIL and not h.alt_rail_available:
        return "NO_ALTERNATE_INSTRUMENT"

    return None


class Simulator:
    def __init__(self, seed: int = 11):
        self._seed = seed

    def _rng(self, txn: Transaction, attempt_no: int) -> random.Random:
        # Per-(txn, attempt) stream so strategies that make the same call at the
        # same point get the same draw. Keeps comparisons fair.
        return random.Random(f"{self._seed}:{txn.txn_id}:{attempt_no}")

    def attempt(
        self, txn: Transaction, action: str, at: datetime, attempt_no: int
    ) -> AttemptResult:
        h = txn.hidden

        if action == GIVE_UP:
            return AttemptResult(action, at, False, 0, 0, "SKIPPED")

        # The customer already paid somewhere else. Charging now is a real
        # incident, not a neutral no-op.
        if h.out_of_band_payment_at and at >= h.out_of_band_payment_at:
            if action in CHARGING_ACTIONS:
                return AttemptResult(
                    action, at, False, 0, COST_RETRY + COST_DOUBLE_CHARGE,
                    "DOUBLE_CHARGE",
                )
            return AttemptResult(action, at, False, 0, COST_LINK, "SKIPPED")

        cost = COST_RETRY if action in CHARGING_ACTIONS else COST_LINK

        blocked = _blocked_reason(txn, action, at)
        if blocked:
            return AttemptResult(action, at, False, 0, cost, "FAIL", blocked)

        base = {
            RETRY_SAME_RAIL: BASE_P_SAME_RAIL,
            RETRY_ALT_RAIL: BASE_P_ALT_RAIL,
            SEND_PAYMENT_LINK: BASE_P_LINK,
        }[action]

        p = base * intent_factor(txn, at)

        # A silent debit on a live mandate does not need the customer present.
        if txn.flow == "subscription" and action in CHARGING_ACTIONS:
            p = base

        rng = self._rng(txn, attempt_no)
        if rng.random() < p:
            return AttemptResult(
                action, at, True, txn.amount_paise, cost, "SUCCESS"
            )
        return AttemptResult(action, at, False, 0, cost, "FAIL", "AUTH_ABANDONED")

    def recoverable_ceiling(self, txn: Transaction) -> bool:
        """
        Was this transaction recoverable at all by an oracle acting optimally?
        Used only for reporting headroom -- no strategy may call this.
        """
        h = txn.hidden
        if h.risk_blocked:
            return False
        if h.out_of_band_payment_at is not None:
            return False
        return True
