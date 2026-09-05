"""
Synthetic corpus of failed payments.

Design note (this is the part that keeps the evaluation honest):

Each transaction is generated from a *hidden root cause* plus hidden state
(when funds arrive, when an outage ends, whether the instrument is really
dead). The agent never sees any of that. It sees only what a real recovery
system would see: the error code, the gateway's free-text message, the rail,
the amount, the flow type, and the customer's prior history.

The simulator (simulator.py) resolves retries against the hidden state. So the
agent is never scored against its own beliefs -- it is scored against a world
model it cannot read. Two error codes (DO_NOT_HONOUR, GATEWAY_TIMEOUT) are
deliberately ambiguous: the same code is emitted by several different root
causes, and the only extra signal is the noisy gateway message. That ambiguity
is where a language model has to earn its place over a lookup table.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict

SIM_START = datetime(2026, 3, 1, 0, 0)
SIM_HORIZON_DAYS = 14

RAILS = ["upi", "card", "netbanking"]
FLOWS = ["checkout", "subscription", "invoice"]

# Root causes -> weight. These are hidden from the agent.
ROOT_CAUSES = {
    "FUNDS": 0.22,
    "OUTAGE": 0.12,
    "LIMIT": 0.06,
    "DEAD_INSTRUMENT": 0.15,
    "RISK_BLOCK": 0.08,
    "USER_DROP": 0.27,
    "TRANSIENT_TECH": 0.10,
}

# Root cause -> (error_code, weight). Note DO_NOT_HONOUR and GATEWAY_TIMEOUT
# appear under multiple causes. That is intentional.
CODE_TABLE: Dict[str, List[tuple]] = {
    "FUNDS": [("INSUFFICIENT_FUNDS", 0.85), ("DO_NOT_HONOUR", 0.15)],
    "OUTAGE": [("ISSUER_DOWN", 0.55), ("GATEWAY_TIMEOUT", 0.45)],
    "LIMIT": [("LIMIT_EXCEEDED", 0.7), ("DO_NOT_HONOUR", 0.3)],
    "DEAD_INSTRUMENT": [
        ("CARD_EXPIRED", 0.22),
        ("CARD_BLOCKED", 0.2),
        ("MANDATE_REVOKED", 0.18),
        ("INVALID_VPA", 0.18),
        ("DO_NOT_HONOUR", 0.22),
    ],
    "RISK_BLOCK": [("RISK_DECLINED_RZP", 0.6), ("DO_NOT_HONOUR", 0.4)],
    "USER_DROP": [("AUTH_ABANDONED", 0.6), ("UPI_APP_TIMEOUT", 0.4)],
    "TRANSIENT_TECH": [("GATEWAY_TIMEOUT", 0.6), ("UPI_APP_TIMEOUT", 0.4)],
}

# Free-text gateway messages. Some carry a genuine hint about the root cause
# that the bare error code does not. A rule table keyed on the code alone
# cannot use these; a language model can.
MESSAGES: Dict[tuple, List[str]] = {
    ("FUNDS", "INSUFFICIENT_FUNDS"): [
        "Insufficient balance in account",
        "Account balance low - transaction declined by issuer",
        "Debit failed: available balance below transaction amount",
    ],
    ("FUNDS", "DO_NOT_HONOUR"): [
        "Do not honour - insufficient funds indicated by issuer",
        "Declined by issuer (51) - balance check failed",
    ],
    ("OUTAGE", "ISSUER_DOWN"): [
        "Issuer bank server unavailable, please retry later",
        "Bank downtime reported, transaction could not be processed",
    ],
    ("OUTAGE", "GATEWAY_TIMEOUT"): [
        "Timed out waiting for issuer VBV server, retry after some time",
        "No response from acquiring bank within timeout window",
    ],
    ("LIMIT", "LIMIT_EXCEEDED"): [
        "Per day transaction limit exceeded for this account",
        "UPI daily limit breached, try after midnight",
    ],
    ("LIMIT", "DO_NOT_HONOUR"): [
        "Do not honour - velocity/daily cap applied by issuer",
        "Declined - transaction count limit reached for the day",
    ],
    ("DEAD_INSTRUMENT", "CARD_EXPIRED"): [
        "Card has expired",
        "Expiry date invalid - card no longer valid",
    ],
    ("DEAD_INSTRUMENT", "CARD_BLOCKED"): [
        "Card blocked by issuing bank",
        "Card reported lost/stolen - permanently declined",
    ],
    ("DEAD_INSTRUMENT", "MANDATE_REVOKED"): [
        "Mandate cancelled by customer at bank",
        "e-mandate revoked, no further debits permitted",
    ],
    ("DEAD_INSTRUMENT", "INVALID_VPA"): [
        "VPA does not exist or has been deregistered",
        "Invalid virtual payment address",
    ],
    ("DEAD_INSTRUMENT", "DO_NOT_HONOUR"): [
        "Do not honour - card restricted by issuer, contact bank",
        "Declined - instrument permanently blocked at issuer",
    ],
    ("RISK_BLOCK", "RISK_DECLINED_RZP"): [
        "Blocked by risk engine - merchant rule match",
        "Transaction declined by internal fraud checks",
    ],
    ("RISK_BLOCK", "DO_NOT_HONOUR"): [
        "Do not honour - suspected fraud flag at issuer",
        "Declined - issuer risk hold on this customer",
    ],
    ("USER_DROP", "AUTH_ABANDONED"): [
        "Customer did not complete OTP authentication",
        "3DS page abandoned by user",
    ],
    ("USER_DROP", "UPI_APP_TIMEOUT"): [
        "Collect request expired - customer did not approve in UPI app",
        "User did not authorise the mandate within window",
    ],
    ("TRANSIENT_TECH", "GATEWAY_TIMEOUT"): [
        "Upstream timeout, no final status received",
        "Temporary technical error at payment processor",
    ],
    ("TRANSIENT_TECH", "UPI_APP_TIMEOUT"): [
        "PSP handle unreachable, temporary failure",
        "Transient error at UPI switch",
    ],
}


@dataclass
class HiddenState:
    """Ground truth. Never exposed to the agent."""
    root_cause: str
    instrument_dead: bool = False
    risk_blocked: bool = False
    funds_available_at: Optional[datetime] = None
    outage_end_at: Optional[datetime] = None
    limit_reset_at: Optional[datetime] = None
    alt_rail_available: bool = True
    intent_half_life_h: float = 12.0
    out_of_band_payment_at: Optional[datetime] = None


@dataclass
class Transaction:
    txn_id: str
    customer_id: str
    amount_paise: int
    rail: str
    flow: str
    failed_at: datetime
    error_code: str
    gateway_message: str
    prior_failures_90d: int
    prior_successes_90d: int
    is_first_time_customer: bool
    hidden: HiddenState = field(repr=False)

    def observable(self) -> dict:
        """Exactly what the agent is allowed to condition on."""
        return {
            "txn_id": self.txn_id,
            "amount_inr": round(self.amount_paise / 100, 2),
            "rail": self.rail,
            "flow": self.flow,
            "failed_at": self.failed_at.isoformat(),
            "error_code": self.error_code,
            "gateway_message": self.gateway_message,
            "prior_failures_90d": self.prior_failures_90d,
            "prior_successes_90d": self.prior_successes_90d,
            "is_first_time_customer": self.is_first_time_customer,
        }


def _weighted(rng: random.Random, pairs) -> str:
    items = [p[0] for p in pairs]
    weights = [p[1] for p in pairs]
    return rng.choices(items, weights=weights, k=1)[0]


# Inflow days. Salary lands on the 1st for most salaried customers; the 7th
# and 15th cover staggered payrolls and freelance cycles; the 25th covers
# advances and UPI credits from family. This is the structure the agent has to
# discover -- it is never told these dates.
INFLOW_DAYS = [1, 7, 15, 25]


def _next_payday(rng: random.Random, t: datetime) -> datetime:
    """When the customer's balance is next topped up."""
    cur = t
    for _ in range(40):
        cur += timedelta(days=1)
        if cur.day in INFLOW_DAYS:
            arrival = cur.replace(
                hour=rng.randint(9, 19), minute=rng.randint(0, 59), second=0
            )
            # A minority are genuinely broke and miss the first inflow.
            if rng.random() < 0.18:
                continue
            return arrival
    return t + timedelta(days=rng.randint(3, 12))


def generate(n: int = 2000, seed: int = 7) -> List[Transaction]:
    rng = random.Random(seed)
    txns: List[Transaction] = []

    for i in range(n):
        cause = _weighted(rng, list(ROOT_CAUSES.items()))
        code = _weighted(rng, CODE_TABLE[cause])
        message = rng.choice(MESSAGES[(cause, code)])

        flow = rng.choices(FLOWS, weights=[0.62, 0.23, 0.15], k=1)[0]
        rail = rng.choices(RAILS, weights=[0.55, 0.36, 0.09], k=1)[0]
        if code in ("CARD_EXPIRED", "CARD_BLOCKED"):
            rail = "card"
        if code == "INVALID_VPA":
            rail = "upi"

        # Amounts: log-normal-ish, subscriptions smaller, invoices larger.
        if flow == "subscription":
            amount = int(rng.lognormvariate(6.4, 0.5)) * 100
        elif flow == "invoice":
            amount = int(rng.lognormvariate(9.4, 0.8)) * 100
        else:
            amount = int(rng.lognormvariate(7.2, 0.9)) * 100
        amount = max(4900, min(amount, 25_000_00))

        failed_at = SIM_START + timedelta(
            minutes=rng.randint(0, SIM_HORIZON_DAYS * 24 * 60)
        )

        h = HiddenState(root_cause=cause)

        if cause == "FUNDS":
            h.funds_available_at = _next_payday(rng, failed_at)
        elif cause == "OUTAGE":
            # Most issuer outages resolve inside a few hours.
            h.outage_end_at = failed_at + timedelta(
                minutes=int(rng.lognormvariate(4.2, 1.0))
            )
        elif cause == "LIMIT":
            reset = (failed_at + timedelta(days=1)).replace(
                hour=0, minute=5, second=0
            )
            h.limit_reset_at = reset
        elif cause == "DEAD_INSTRUMENT":
            h.instrument_dead = True
        elif cause == "RISK_BLOCK":
            # A minority of risk holds lift on their own.
            if rng.random() < 0.15:
                h.outage_end_at = failed_at + timedelta(hours=rng.randint(12, 72))
            else:
                h.risk_blocked = True
        elif cause == "TRANSIENT_TECH":
            h.outage_end_at = failed_at + timedelta(minutes=rng.randint(1, 45))
        # USER_DROP: nothing is broken. The customer simply has to come back.

        h.alt_rail_available = rng.random() < 0.78
        if flow == "checkout":
            h.intent_half_life_h = rng.uniform(4, 30)
        elif flow == "invoice":
            h.intent_half_life_h = rng.uniform(120, 400)
        else:
            h.intent_half_life_h = 1e6  # mandate exists; intent is not the issue

        # ~6% of customers pay through some other channel on their own.
        # Any charge attempt after this instant is a double charge.
        if rng.random() < 0.06:
            h.out_of_band_payment_at = failed_at + timedelta(
                hours=rng.uniform(1, 96)
            )

        prior_f = rng.choices([0, 1, 2, 3, 6], weights=[0.5, 0.25, 0.13, 0.08, 0.04])[0]
        prior_s = rng.choices([0, 1, 4, 12], weights=[0.28, 0.3, 0.3, 0.12])[0]

        txns.append(
            Transaction(
                txn_id=f"pay_{i:05d}",
                customer_id=f"cust_{rng.randint(0, n // 2):05d}",
                amount_paise=amount,
                rail=rail,
                flow=flow,
                failed_at=failed_at,
                error_code=code,
                gateway_message=message,
                prior_failures_90d=prior_f,
                prior_successes_90d=prior_s,
                is_first_time_customer=(prior_s == 0 and prior_f == 0),
                hidden=h,
            )
        )

    return txns


def summarise(txns: List[Transaction]) -> dict:
    from collections import Counter
    return {
        "n": len(txns),
        "at_risk_inr": round(sum(t.amount_paise for t in txns) / 100, 2),
        "by_root_cause": dict(Counter(t.hidden.root_cause for t in txns)),
        "by_error_code": dict(Counter(t.error_code for t in txns)),
        "by_flow": dict(Counter(t.flow for t in txns)),
        "out_of_band_payers": sum(
            1 for t in txns if t.hidden.out_of_band_payment_at
        ),
    }
