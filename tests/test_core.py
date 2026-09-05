"""Tests for the properties that the results table depends on being true."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datetime import timedelta

from recovery.agent import Gatekeeper, MAX_ACTIONS_PER_TXN, RecoveryAgent
from recovery.corpus import generate
from recovery.diagnoser import RuleDiagnoser, OracleDiagnoser, RISK_TERMINAL
from recovery.simulator import (
    Simulator, RETRY_SAME_RAIL, SEND_PAYMENT_LINK, CHARGING_ACTIONS,
)


def test_agent_never_charges_a_risk_blocked_customer():
    txns = generate(600, seed=2)
    sim = Simulator()
    agent = RecoveryAgent(OracleDiagnoser(), sim)
    results, _ = agent.run_batch(txns)
    blocked = {t.txn_id for t in txns if t.hidden.risk_blocked}
    for r in results:
        if r.txn_id in blocked:
            executed = [e for e in r.audit if e.decision == "EXECUTED"]
            assert all(e.proposed_action not in CHARGING_ACTIONS for e in executed)


def test_action_budget_is_never_exceeded():
    txns = generate(600, seed=3)
    agent = RecoveryAgent(RuleDiagnoser(), Simulator())
    results, _ = agent.run_batch(txns)
    assert all(r.actions_taken <= MAX_ACTIONS_PER_TXN for r in results)


def test_no_customer_contact_during_quiet_hours():
    txns = generate(800, seed=4)
    agent = RecoveryAgent(RuleDiagnoser(), Simulator())
    results, _ = agent.run_batch(txns)
    for r in results:
        for e in r.audit:
            if e.decision == "EXECUTED" and e.proposed_action == SEND_PAYMENT_LINK:
                hour = int(e.at[11:13])
                assert 8 <= hour < 21, e


def test_agent_stops_after_success():
    txns = generate(400, seed=5)
    agent = RecoveryAgent(RuleDiagnoser(), Simulator())
    results, _ = agent.run_batch(txns)
    for r in results:
        outcomes = [e.outcome for e in r.audit if e.decision == "EXECUTED"]
        if "SUCCESS" in outcomes:
            assert outcomes.index("SUCCESS") == len(outcomes) - 1


def test_simulator_is_deterministic():
    txns = generate(50, seed=6)
    a = Simulator(seed=1).attempt(txns[0], RETRY_SAME_RAIL, txns[0].failed_at + timedelta(hours=3), 1)
    b = Simulator(seed=1).attempt(txns[0], RETRY_SAME_RAIL, txns[0].failed_at + timedelta(hours=3), 1)
    assert a.success == b.success and a.cost_paise == b.cost_paise


def test_every_executed_action_has_an_audit_entry_with_a_reason():
    txns = generate(300, seed=8)
    agent = RecoveryAgent(RuleDiagnoser(), Simulator())
    results, _ = agent.run_batch(txns)
    for r in results:
        for e in r.audit:
            assert e.reason and e.decision in ("EXECUTED", "BLOCKED")


def test_gatekeeper_blocks_contact_fatigue():
    txns = generate(10, seed=9)
    t = txns[0]
    g = Gatekeeper()
    at = t.failed_at.replace(hour=10)
    for _ in range(2):
        assert g.check(t, SEND_PAYMENT_LINK, at) is None
        g.record(SEND_PAYMENT_LINK, at)
        at += timedelta(hours=25)
    assert g.check(t, SEND_PAYMENT_LINK, at) is not None


def test_actions_are_spaced_and_strictly_ordered():
    """No two money actions on one payment may leave at the same instant."""
    from recovery.agent import MIN_GAP_BETWEEN_ACTIONS_MIN
    from datetime import datetime
    txns = generate(800, seed=11)
    agent = RecoveryAgent(RuleDiagnoser(), Simulator())
    results, _ = agent.run_batch(txns)
    for r in results:
        times = [datetime.fromisoformat(e.at) for e in r.audit
                 if e.decision == "EXECUTED" and e.proposed_action != "GIVE_UP"]
        for a, b in zip(times, times[1:]):
            gap = (b - a).total_seconds() / 60
            assert gap >= MIN_GAP_BETWEEN_ACTIONS_MIN - 1e-6, (a, b)