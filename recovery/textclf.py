"""
A free, local alternative to the LLM diagnoser.

The rule table scores 0% on DO_NOT_HONOUR because a lookup keyed on the error
code cannot read the gateway message. An LLM can read it. So can a naive Bayes
classifier over the message text, and that costs nothing and runs offline.

The interesting part is where the training labels come from. We do not use the
hidden root cause -- that would be cheating, and in production it does not
exist. Instead we recover weak labels from *what actually worked* on the train
split:

    same-rail retry succeeds early          -> nothing was broken
    same-rail fails, alternate rail works   -> the instrument was the problem
    early attempts fail, a late one works   -> it was a timing problem
    nothing ever works, on any rail, ever   -> terminal; stop

That is exactly the signal a merchant's retry log already contains. It is
noisy, and it is defined by outcome rather than by cause -- a daily-limit
failure and a dead card both look like "alternate rail works", so they collapse
into one label. That costs us reported accuracy against the hidden cause but
loses nothing operationally, because both take the same action. See the README.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List

from .corpus import Transaction
from .diagnoser import (
    AMBIGUOUS_CODES,
    Diagnosis,
    FUNDS_TIMED,
    INSTRUMENT_DEAD,
    RISK_TERMINAL,
    RuleDiagnoser,
    TRANSIENT_SHORT,
    UNKNOWN,
    USER_INTENT_DROP,
)
from .simulator import (
    RETRY_ALT_RAIL,
    RETRY_SAME_RAIL,
    SEND_PAYMENT_LINK,
    Simulator,
)

# A fixed diagnostic battery. Each probe answers one question.
PROBES = [
    ("early_same", RETRY_SAME_RAIL, 2.0),
    ("early_alt", RETRY_ALT_RAIL, 2.0),
    ("early_link", SEND_PAYMENT_LINK, 2.0),
    ("mid_same", RETRY_SAME_RAIL, 72.0),
    ("late_same", RETRY_SAME_RAIL, 312.0),
    ("late_link", SEND_PAYMENT_LINK, 312.0),
]

TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(txn: Transaction) -> List[str]:
    words = TOKEN_RE.findall(txn.gateway_message.lower())
    grams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + grams + [f"CODE={txn.error_code}", f"RAIL={txn.rail}"]


def weak_label(outcomes: Dict[str, bool]) -> str:
    """Label a train transaction from what worked, not from why it failed."""
    if outcomes["early_same"]:
        return USER_INTENT_DROP          # nothing was broken; act immediately
    if outcomes["early_alt"] or outcomes["early_link"]:
        return INSTRUMENT_DEAD           # route around this instrument now
    if outcomes["mid_same"] or outcomes["late_same"] or outcomes["late_link"]:
        return FUNDS_TIMED               # it was a timing problem; wait
    return RISK_TERMINAL                 # nothing ever worked; stop


class NaiveBayes:
    """Multinomial NB with Laplace smoothing. ~40 lines, no dependencies."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.tok_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int))
        self.class_total: Dict[str, int] = defaultdict(int)
        self.vocab: set = set()

    def fit(self, docs: List[List[str]], labels: List[str]) -> None:
        for toks, y in zip(docs, labels):
            self.class_counts[y] += 1
            for t in toks:
                self.tok_counts[y][t] += 1
                self.class_total[y] += 1
                self.vocab.add(t)

    def predict(self, toks: List[str]) -> tuple:
        n = sum(self.class_counts.values())
        v = len(self.vocab)
        scores = {}
        for y, cnt in self.class_counts.items():
            lp = math.log(cnt / n)
            denom = self.class_total[y] + self.alpha * v
            for t in toks:
                lp += math.log((self.tok_counts[y][t] + self.alpha) / denom)
            scores[y] = lp
        best = max(scores, key=scores.get)
        top = scores[best]
        z = sum(math.exp(s - top) for s in scores.values())
        return best, 1.0 / z


class LocalTextDiagnoser:
    """
    Rules for the codes that mean one thing; the trained classifier for the
    three that don't. Same routing logic as HybridDiagnoser, no API, no cost.
    """

    name = "local_clf"

    def __init__(self, model: NaiveBayes, train_size: int = 0):
        self.model = model
        self.rules = RuleDiagnoser()
        self.train_size = train_size
        self.routed_to_clf = 0
        self.routed_to_rules = 0

    def diagnose_batch(self, txns: List[Transaction]) -> List[Diagnosis]:
        out = []
        for t in txns:
            if t.error_code not in AMBIGUOUS_CODES:
                self.routed_to_rules += 1
                out.append(self.rules.diagnose_batch([t])[0])
                continue
            self.routed_to_clf += 1
            cls, conf = self.model.predict(tokenize(t))
            wait = {TRANSIENT_SHORT: 1.0, USER_INTENT_DROP: 0.25,
                    FUNDS_TIMED: 72.0, INSTRUMENT_DEAD: 0.5,
                    RISK_TERMINAL: 0.0}.get(cls, 4.0)
            out.append(Diagnosis(
                cls, round(conf, 3), wait,
                f"nb p={conf:.2f} on gateway message", "local_clf"))
        return out


def fit(train: List[Transaction], sim: Simulator) -> tuple:
    """Run the probe battery on the train split and fit the classifier."""
    docs, labels = [], []
    for txn in train:
        outcomes = {}
        for i, (name, action, hours) in enumerate(PROBES):
            from datetime import timedelta
            res = sim.attempt(txn, action, txn.failed_at + timedelta(hours=hours),
                              attempt_no=200 + i)
            outcomes[name] = res.outcome == "SUCCESS"
        docs.append(tokenize(txn))
        labels.append(weak_label(outcomes))

    nb = NaiveBayes()
    nb.fit(docs, labels)
    return LocalTextDiagnoser(nb, len(train)), labels


def label_distribution(labels: List[str]) -> dict:
    from collections import Counter
    return dict(Counter(labels))