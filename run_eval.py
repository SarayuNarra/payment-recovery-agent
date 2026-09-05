#!/usr/bin/env python3
"""
Benchmark entrypoint.

    python run_eval.py --n 4000 --seed 7
    ANTHROPIC_API_KEY=... python run_eval.py --llm

The corpus is split 40/60. The 40% train split is used only to fit the timing
model (recovery/learn.py). Every number in the results table is computed on the
untouched 60% test split.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from recovery import learn
from recovery.agent import RecoveryAgent, plan_actions
from recovery.corpus import generate, summarise
from recovery.diagnoser import (
    HybridDiagnoser, LLMDiagnoser, OracleDiagnoser, RuleDiagnoser,
    diagnosis_accuracy,
)
from recovery.report import write_html
from recovery import textclf
from recovery.evaluate import build_baselines, score
from recovery.simulator import Simulator

RESULTS = Path(__file__).parent / "results"

COLUMNS = [
    ("strategy", "Strategy", "s"),
    ("recovered_inr", "Recovered ₹", "money"),
    ("recovery_rate_value", "Rec. rate", "pct"),
    ("share_of_ceiling", "% of ceiling", "pct"),
    ("net_inr", "Net ₹", "money"),
    ("cost_per_rupee_recovered", "Cost/₹ rec.", "ratio"),
    ("actions_per_txn", "Actions/txn", "num"),
    ("double_charges", "Double charges", "int"),
    ("wasted_attempts_on_dead_instruments", "Wasted attempts", "int"),
]


def _cell(v, kind):
    if v is None:
        return "—"
    if kind == "pct":
        return f"{v*100:.1f}%"
    if kind == "money":
        return f"{v:,.0f}"
    if kind == "ratio":
        return f"{v:.4f}"
    if kind == "num":
        return f"{v:.2f}"
    if kind == "int":
        return f"{v:,}"
    return str(v)


def table(rows):
    head = "| " + " | ".join(c[1] for c in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = ["| " + " | ".join(_cell(r[k], t) for k, _, t in COLUMNS) + " |"
            for r in rows]
    return "\n".join([head, sep] + body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--llm", action="store_true",
                    help="add the LLM arm (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--strict", action="store_true",
                    help="crash on the first API error instead of falling back "
                         "to the rule table (use this to debug the LLM arm)")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)

    txns = generate(args.n, seed=args.seed)
    split = int(len(txns) * 0.4)
    train, test = txns[:split], txns[split:]

    sim = Simulator(seed=args.seed + 4)
    rules = RuleDiagnoser()

    print(json.dumps(summarise(test), indent=2))
    print(f"\ntrain={len(train)}  test={len(test)}\n")

    timing = learn.fit(train, rules, sim)
    print(f"timing model fitted on {timing.n_probes:,} probe outcomes "
          f"from the train split\n")
    for dc in ("FUNDS_TIMED", "USER_INTENT_DROP", "INSTRUMENT_DEAD"):
        print(learn.report(timing, dc))
    print()

    local_clf, weak = textclf.fit(train, sim)
    print(f"local text classifier fitted on {len(train):,} weakly-labelled "
          f"train rows: {textclf.label_distribution(weak)}\n")

    strategies = list(build_baselines(sim).items())
    strategies.append(
        ("agent_rules_handwritten", RecoveryAgent(rules, sim, "agent_rules")))
    strategies.append(
        ("agent_rules_learned",
         RecoveryAgent(rules, sim, "agent_learned", timing_model=timing)))

    strategies.append(
        ("agent_localclf_learned",
         RecoveryAgent(local_clf, sim, "agent_localclf", timing_model=timing)))

    hybrid = None
    if args.llm:
        # Hybrid routing: only the ambiguous error codes go to the model.
        hybrid = HybridDiagnoser(model=args.model, strict=args.strict)
        strategies.append(
            ("agent_llm_learned",
             RecoveryAgent(hybrid, sim, "agent_llm", timing_model=timing)))

    strategies.append(
        ("[upper bound] agent_oracle_learned",
         RecoveryAgent(OracleDiagnoser(), sim, "oracle", timing_model=timing)))

    HEADLINE = "agent_localclf_learned"
    rows, diag_report, audit_sample = [], {}, []
    for name, strat in strategies:
        results, diags = strat.run_batch(test)
        row = score(test, results, sim)
        row["strategy"] = name
        rows.append(row)
        if diags is not None:
            diag_report[name] = diagnosis_accuracy(test, diags)
        # Audit comes from the headline strategy only. Collecting from several
        # arms and grouping by txn_id interleaves their logs and makes a single
        # payment look like it was actioned twice.
        if name == HEADLINE:
            for r in results[:40]:
                audit_sample.extend(asdict(e) for e in r.audit)

    print(table(rows))

    if hybrid is not None:
        llm = hybrid.llm
        print(f"\nLLM routing: {hybrid.routed_to_llm:,} txns to the model, "
              f"{hybrid.routed_to_rules:,} to the rule table")
        print(f"LLM: {llm.unique_questions} distinct questions after dedup "
              f"({hybrid.routed_to_llm:,} rows collapsed), "
              f"{llm.cache_hits:,} served from disk cache")
        print(f"LLM: {llm.calls} API calls, {llm.fallbacks:,} txns fell back")
        if llm.fallbacks:
            print("\n  !! THE LLM ARM DID NOT RUN CLEANLY.")
            print("  !! Its row is the rule table's row. Do not report it.")
            for e in dict.fromkeys(llm.errors):
                print(f"     {e}")
            print("  !! Re-run with --strict for the full traceback.\n")

    seen = set()
    for name, rep in diag_report.items():
        key = rep["accuracy"]
        if key in seen:
            continue
        seen.add(key)
        print(f"\n{name}: diagnosis accuracy {rep['accuracy']*100:.1f}%  "
              f"(missed terminal {rep['terminal_missed']}, "
              f"false terminal {rep['terminal_false_positive']})")
        for code, acc in rep["per_error_code"].items():
            print(f"    {code:<22} {acc*100:5.1f}%")

    metrics_blob = (
        {"corpus_test": summarise(test), "strategies": rows,
         "diagnosis": diag_report})
    (RESULTS / "metrics.json").write_text(
        json.dumps(metrics_blob, indent=2), encoding="utf-8")
    (RESULTS / "metrics.md").write_text(
        f"# Results\n\nHeld-out test split: {len(test)} failed payments "
        f"(seed {args.seed}), "
        f"₹{summarise(test)['at_risk_inr']:,.0f} at risk.\n\n"
        + table(rows) + "\n", encoding="utf-8")
    (RESULTS / "audit_sample.json").write_text(
        json.dumps(audit_sample, indent=2), encoding="utf-8")
    write_html(RESULTS / "report.html", metrics_blob, audit_sample,
               headline=HEADLINE)
    print(f"\nWrote {RESULTS}/report.html  <- open this one in a browser")
    print(f"      {RESULTS}/metrics.md, metrics.json, audit_sample.json")


if __name__ == "__main__":
    main()