"""
Diagnosis: turn an observed failure into a hypothesis about why it failed.

Two implementations, deliberately:

  RuleDiagnoser  - a lookup table on the error code. This is what a competent
                   engineer writes without any AI. It is the control.
  LLMDiagnoser   - reads the free-text gateway message and customer history
                   alongside the code, and returns a class + confidence +
                   suggested wait.

The point of shipping both is that the LLM has to *earn* its place. run_eval.py
reports diagnosis accuracy and recovered rupees for both, so the ablation is
visible rather than asserted. If the LLM does not beat the table, that is a
finding, and it belongs in the README.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

from .corpus import Transaction

# Diagnosis classes. These map onto interventions, not onto error codes.
TRANSIENT_SHORT = "TRANSIENT_SHORT"      # retry in minutes to hours
FUNDS_TIMED = "FUNDS_TIMED"              # retry when money arrives
LIMIT_TIMED = "LIMIT_TIMED"              # retry after the daily cap resets
INSTRUMENT_DEAD = "INSTRUMENT_DEAD"      # this instrument will never work again
RISK_TERMINAL = "RISK_TERMINAL"          # do not retry, ever
USER_INTENT_DROP = "USER_INTENT_DROP"    # nothing is broken; re-engage fast
UNKNOWN = "UNKNOWN"

# Ground-truth root cause -> the class we would call correct.
CAUSE_TO_CLASS = {
    "FUNDS": FUNDS_TIMED,
    "OUTAGE": TRANSIENT_SHORT,
    "LIMIT": LIMIT_TIMED,
    "DEAD_INSTRUMENT": INSTRUMENT_DEAD,
    "RISK_BLOCK": RISK_TERMINAL,
    "USER_DROP": USER_INTENT_DROP,
    "TRANSIENT_TECH": TRANSIENT_SHORT,
}


@dataclass
class Diagnosis:
    dclass: str
    confidence: float
    suggested_wait_h: float
    rationale: str
    source: str = "rule"


class RuleDiagnoser:
    """Code -> class lookup. Cannot disambiguate DO_NOT_HONOUR at all."""

    name = "rules"

    TABLE = {
        "INSUFFICIENT_FUNDS": (FUNDS_TIMED, 0.9, 72.0),
        "ISSUER_DOWN": (TRANSIENT_SHORT, 0.9, 1.0),
        "GATEWAY_TIMEOUT": (TRANSIENT_SHORT, 0.7, 0.5),
        "UPI_APP_TIMEOUT": (USER_INTENT_DROP, 0.6, 0.25),
        "LIMIT_EXCEEDED": (LIMIT_TIMED, 0.9, 24.0),
        "CARD_EXPIRED": (INSTRUMENT_DEAD, 0.95, 0.0),
        "CARD_BLOCKED": (INSTRUMENT_DEAD, 0.95, 0.0),
        "MANDATE_REVOKED": (INSTRUMENT_DEAD, 0.95, 0.0),
        "INVALID_VPA": (INSTRUMENT_DEAD, 0.9, 0.0),
        "RISK_DECLINED_RZP": (RISK_TERMINAL, 0.95, 0.0),
        "AUTH_ABANDONED": (USER_INTENT_DROP, 0.85, 0.25),
        # The ambiguous one. The table has to guess, and it guesses the mode.
        "DO_NOT_HONOUR": (UNKNOWN, 0.35, 6.0),
    }

    def diagnose_batch(self, txns: List[Transaction]) -> List[Diagnosis]:
        out = []
        for t in txns:
            cls, conf, wait = self.TABLE.get(t.error_code, (UNKNOWN, 0.3, 6.0))
            out.append(
                Diagnosis(cls, conf, wait, f"code table: {t.error_code}", "rule")
            )
        return out


SYSTEM_PROMPT = """You are a payment failure triage engine for an Indian payment gateway.

For each failed transaction you are given the error code, the raw gateway
message, the rail, the flow type, the amount, and the customer's history.
Classify the underlying cause into exactly one class:

TRANSIENT_SHORT  - a temporary technical or issuer-side fault that clears on its
                   own within minutes to a few hours.
FUNDS_TIMED      - the customer does not have the money right now but will.
LIMIT_TIMED      - a per-day cap or velocity limit that resets at midnight.
INSTRUMENT_DEAD  - this specific instrument or mandate is permanently unusable.
                   A different instrument would work.
RISK_TERMINAL    - blocked by fraud or risk controls. Never retry.
USER_INTENT_DROP - nothing is broken. The customer simply did not complete
                   authentication.
UNKNOWN          - genuinely cannot tell.

The error code alone is often insufficient. DO_NOT_HONOUR and GATEWAY_TIMEOUT in
particular are emitted for several different underlying causes; the gateway
message usually contains the distinguishing detail. Read it carefully.

Return ONLY a JSON array, one object per transaction, in the same order:
[{"txn_id": "...", "class": "...", "confidence": 0.0-1.0,
  "suggested_wait_hours": <number>, "rationale": "<max 15 words>"}]

No prose, no markdown fences."""


class LLMDiagnoser:
    """
    Calls the Anthropic messages API in batches.

    Falls back to RuleDiagnoser for any batch that errors or returns malformed
    JSON, so a network failure degrades the numbers rather than crashing the
    run. Every fallback is counted and reported.
    """

    name = "llm"

    def __init__(self, model: str = "claude-haiku-4-5-20251001",
                 batch_size: int = 25, strict: bool = False):
        self.model = model
        self.batch_size = batch_size
        self.strict = strict
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.fallbacks = 0
        self.calls = 0
        self.errors: list = []
        self.cache_hits = 0
        self.unique_questions = 0
        self._rules = RuleDiagnoser()

    def _call(self, payload: List[dict]) -> Optional[list]:
        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": 4000,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": json.dumps(payload, indent=None)}
                ],
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key or "",
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            # The API puts the actual reason in the response body. Without
            # reading it you get "HTTP Error 404" and no idea why.
            detail = e.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"network: {e.reason}") from None
        text = "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        return json.loads(text)

    # --- deduplication -------------------------------------------------
    # Diagnosis depends on the error code and the gateway message. It does not
    # depend on the amount or the transaction id. Across 915 ambiguous rows
    # there are only ~100 distinct questions, so ask each one once, cache the
    # answer to disk, and reuse it. This is not a hackathon trick -- it is what
    # you would do in production, where the same twelve message templates
    # account for most of daily volume.

    CACHE_PATH = Path(__file__).resolve().parents[1] / "results" / "llm_cache.json"

    @staticmethod
    def _key(t: Transaction) -> str:
        bucket = "0" if t.prior_failures_90d == 0 else (
            "1-2" if t.prior_failures_90d <= 2 else "3+")
        return "|".join([t.error_code, t.gateway_message, t.flow, bucket])

    def _load_cache(self) -> dict:
        if self.CACHE_PATH.exists():
            try:
                return json.loads(self.CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self, cache: dict) -> None:
        self.CACHE_PATH.parent.mkdir(exist_ok=True)
        self.CACHE_PATH.write_text(
            json.dumps(cache, indent=2), encoding="utf-8")

    def diagnose_batch(self, txns: List[Transaction]) -> List[Diagnosis]:
        if not self.api_key:
            msg = "ANTHROPIC_API_KEY is not set"
            self.errors.append(msg)
            if self.strict:
                raise RuntimeError(msg)
            self.fallbacks += len(txns)
            return self._rules.diagnose_batch(txns)

        cache = self._load_cache()
        self.cache_hits = sum(1 for t in txns if self._key(t) in cache)

        # One representative transaction per distinct question.
        reps: dict = {}
        for t in txns:
            k = self._key(t)
            if k not in cache and k not in reps:
                reps[k] = t
        self.unique_questions = len(reps)
        pending = list(reps.items())

        for i in range(0, len(pending), self.batch_size):
            chunk_kv = pending[i : i + self.batch_size]
            chunk = [t for _, t in chunk_kv]
            payload = [
                {
                    "txn_id": t.txn_id,
                    "error_code": t.error_code,
                    "gateway_message": t.gateway_message,
                    "rail": t.rail,
                    "flow": t.flow,
                    "prior_failures_90d": t.prior_failures_90d,
                }
                for t in chunk
            ]
            try:
                self.calls += 1
                parsed = self._call(payload)
                if not isinstance(parsed, list) or len(parsed) != len(chunk):
                    raise ValueError("shape mismatch")
                for (k, _), item in zip(chunk_kv, parsed):
                    cache[k] = {
                        "class": item.get("class", UNKNOWN),
                        "confidence": float(item.get("confidence", 0.5)),
                        "suggested_wait_hours": float(
                            item.get("suggested_wait_hours", 2.0)),
                        "rationale": str(item.get("rationale", ""))[:120],
                    }
            except Exception as exc:  # noqa: BLE001 - degrade, do not crash
                self.errors.append(f"{type(exc).__name__}: {exc}")
                if self.strict:
                    raise
                for (k, t), d in zip(chunk_kv, self._rules.diagnose_batch(chunk)):
                    cache[k] = {
                        "class": d.dclass, "confidence": d.confidence,
                        "suggested_wait_hours": d.suggested_wait_h,
                        "rationale": f"fallback ({type(exc).__name__})",
                        "_fallback": True,
                    }

        self._save_cache(cache)

        out: List[Diagnosis] = []
        for t in txns:
            c = cache.get(self._key(t))
            if c is None:
                self.fallbacks += 1
                out.extend(self._rules.diagnose_batch([t]))
                continue
            if c.get("_fallback"):
                self.fallbacks += 1
            out.append(Diagnosis(
                c["class"], c["confidence"], c["suggested_wait_hours"],
                c["rationale"],
                "rule-fallback" if c.get("_fallback") else "llm"))
        return out


def diagnosis_accuracy(txns: List[Transaction], diags: List[Diagnosis]) -> dict:
    """Scored against the hidden root cause the diagnoser never saw."""
    total = len(txns)
    correct = 0
    terminal_missed = 0   # called retryable, actually RISK_BLOCK
    terminal_false = 0    # called RISK_TERMINAL, actually recoverable
    per_code: dict = {}

    for t, d in zip(txns, diags):
        truth = CAUSE_TO_CLASS[t.hidden.root_cause]
        ok = d.dclass == truth
        correct += ok
        bucket = per_code.setdefault(t.error_code, [0, 0])
        bucket[0] += ok
        bucket[1] += 1
        if truth == RISK_TERMINAL and d.dclass != RISK_TERMINAL:
            terminal_missed += 1
        if d.dclass == RISK_TERMINAL and truth != RISK_TERMINAL:
            terminal_false += 1

    return {
        "accuracy": round(correct / total, 4),
        "terminal_missed": terminal_missed,
        "terminal_false_positive": terminal_false,
        "per_error_code": {
            k: round(v[0] / v[1], 3) for k, v in sorted(per_code.items())
        },
    }


class OracleDiagnoser:
    """
    Reads the hidden root cause directly. NOT a strategy -- it cannot exist in
    production. It is in the benchmark to answer one question: how much of the
    remaining gap is a diagnosis problem versus a world that simply cannot be
    recovered? Without this row, "82% accuracy" has no scale.
    """

    name = "oracle"

    def diagnose_batch(self, txns: List[Transaction]) -> List[Diagnosis]:
        return [
            Diagnosis(CAUSE_TO_CLASS[t.hidden.root_cause], 1.0, 0.0,
                      "oracle: hidden root cause", "oracle")
            for t in txns
        ]


AMBIGUOUS_CODES = {"DO_NOT_HONOUR", "GATEWAY_TIMEOUT", "UPI_APP_TIMEOUT"}


class HybridDiagnoser:
    """
    Route only the codes the rule table cannot resolve to the model.

    On this corpus the ambiguous codes are ~29% of volume, so this cuts LLM
    spend by roughly 70% while keeping essentially all of the accuracy gain.
    A code like MANDATE_REVOKED means exactly one thing; paying a model to
    restate a lookup table is waste, and waste at 4,000 rows/day is a real
    line item.
    """

    name = "hybrid"

    def __init__(self, llm: Optional["LLMDiagnoser"] = None, **llm_kwargs):
        self.rules = RuleDiagnoser()
        self.llm = llm or LLMDiagnoser(**llm_kwargs)
        self.routed_to_llm = 0
        self.routed_to_rules = 0

    def diagnose_batch(self, txns: List[Transaction]) -> List[Diagnosis]:
        idx_llm = [i for i, t in enumerate(txns) if t.error_code in AMBIGUOUS_CODES]
        idx_rule = [i for i, t in enumerate(txns) if t.error_code not in AMBIGUOUS_CODES]
        self.routed_to_llm += len(idx_llm)
        self.routed_to_rules += len(idx_rule)

        out: List[Optional[Diagnosis]] = [None] * len(txns)
        for i, d in zip(idx_rule, self.rules.diagnose_batch([txns[i] for i in idx_rule])):
            out[i] = d
        if idx_llm:
            for i, d in zip(idx_llm, self.llm.diagnose_batch([txns[i] for i in idx_llm])):
                out[i] = d
        return [d for d in out if d is not None]