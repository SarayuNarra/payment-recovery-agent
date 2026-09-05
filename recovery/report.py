"""
Renders results/report.html from the metrics blob.

Deliberately a static, self-contained file: no server, no build step, no CDN.
Open it with a double-click. The audit trail viewer at the bottom is the part
worth showing to a reviewer -- it makes "every money action is explainable"
something you can click through rather than a claim in a README.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

CSS = """
:root{--bg:#0e1116;--panel:#161b22;--line:#242c37;--fg:#e6edf3;--dim:#8b949e;
--good:#3fb950;--warn:#d29922;--bad:#f85149;--accent:#58a6ff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
padding:32px 24px 80px;}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:17px;margin:38px 0 12px;color:var(--fg)}
.sub{color:var(--dim);margin:0 0 26px;font-size:14px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.card .k{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:26px;font-weight:600;margin-top:6px}
.card .n{color:var(--dim);font-size:12px;margin-top:4px}
table{width:100%;border-collapse:collapse;background:var(--panel);
border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:14px}
th{text-align:left;padding:10px 12px;color:var(--dim);font-weight:500;
border-bottom:1px solid var(--line);font-size:12px;text-transform:uppercase;
letter-spacing:.05em}
td{padding:10px 12px;border-bottom:1px solid var(--line)}
tr:last-child td{border-bottom:none}
tr.hero td{background:#16241a}
tr.bound td{color:var(--dim);font-style:italic}
.bar{position:relative;height:22px;background:#1e2530;border-radius:4px;min-width:150px}
.bar i{position:absolute;inset:0 auto 0 0;background:var(--accent);border-radius:4px;opacity:.55}
.bar span{position:relative;padding-left:8px;line-height:22px;font-variant-numeric:tabular-nums}
tr.hero .bar i{background:var(--good);opacity:.7}
.num{font-variant-numeric:tabular-nums;text-align:right}
.note{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:8px;padding:14px 16px;color:var(--dim);font-size:14px;margin:14px 0}
.note.alert{border-left-color:var(--bad);color:#ffb4ae}
.acc{display:flex;flex-wrap:wrap;gap:8px}
.pill{background:var(--panel);border:1px solid var(--line);border-radius:20px;
padding:5px 12px;font-size:13px}
.pill b{font-variant-numeric:tabular-nums}
.ok{color:var(--good)}.mid{color:var(--warn)}.no{color:var(--bad)}
select{background:var(--panel);color:var(--fg);border:1px solid var(--line);
border-radius:8px;padding:9px 12px;font-size:14px;min-width:280px}
.steps{margin-top:14px;display:flex;flex-direction:column;gap:8px}
.step{background:var(--panel);border:1px solid var(--line);border-radius:9px;
padding:12px 14px;display:grid;grid-template-columns:78px 1fr;gap:14px}
.step.blocked{opacity:.62;border-style:dashed}
.tag{font-size:11px;font-weight:600;letter-spacing:.05em;padding:3px 0}
.tag.EXECUTED{color:var(--accent)}.tag.BLOCKED{color:var(--warn)}
.step .a{font-weight:600}
.step .r{color:var(--dim);font-size:13px;margin-top:3px}
.step .m{font-size:13px;margin-top:5px;font-variant-numeric:tabular-nums}
.SUCCESS{color:var(--good)}.FAIL{color:var(--dim)}.DOUBLE_CHARGE{color:var(--bad)}
"""


def _inr(v: float) -> str:
    return f"₹{v:,.0f}"


def write_html(path: Path, metrics: dict, audit: list,
               headline: str = "") -> None:
    rows = metrics["strategies"]
    corpus = metrics["corpus_test"]
    diag = metrics.get("diagnosis", {})

    best_agent = max(
        (r for r in rows if r["strategy"].startswith("agent")),
        key=lambda r: r["recovered_inr"],
    )
    best_naive = max(
        (r for r in rows if not r["strategy"].startswith(("agent", "[upper"))),
        key=lambda r: r["recovered_inr"],
    )
    lift = best_agent["recovered_inr"] - best_naive["recovered_inr"]
    peak = max(r["recovered_inr"] for r in rows) or 1

    cards = [
        ("At risk", _inr(corpus["at_risk_inr"]), f"{corpus['n']:,} failed payments"),
        ("Recovered by agent", _inr(best_agent["recovered_inr"]),
         f"{best_agent['recovery_rate_value']*100:.1f}% of value at risk"),
        ("Lift over best baseline", "+" + _inr(lift),
         f"vs {best_naive['strategy'].replace('_',' ')}"),
        ("Attempts per payment", f"{best_agent['actions_per_txn']:.2f}",
         f"vs {best_naive['actions_per_txn']:.2f} — fewer, not more"),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="n">{n}</div></div>' for k, v, n in cards
    )

    trs = []
    for r in rows:
        name = r["strategy"]
        cls = ""
        if name == best_agent["strategy"]:
            cls = "hero"
        elif name.startswith("[upper"):
            cls = "bound"
        pct = r["recovered_inr"] / peak * 100
        label = name.replace("[upper bound] ", "").replace("_", " ")
        if cls == "bound":
            label += "  (not a strategy — ceiling on diagnosis quality)"
        trs.append(
            f'<tr class="{cls}"><td>{label}</td>'
            f'<td><div class="bar"><i style="width:{pct:.1f}%"></i>'
            f'<span>{_inr(r["recovered_inr"])}</span></div></td>'
            f'<td class="num">{r["recovery_rate_value"]*100:.1f}%</td>'
            f'<td class="num">{r["share_of_ceiling"]*100:.1f}%</td>'
            f'<td class="num">{r["actions_per_txn"]:.2f}</td>'
            f'<td class="num">{r["double_charges"]}</td>'
            f'<td class="num">{r["wasted_attempts_on_dead_instruments"]:,}</td></tr>'
        )

    # Diagnosis accuracy for the non-oracle arm.
    # Show the headline strategy's diagnosis, not whichever arm happened to be
    # first in the dict.
    pills = accuracy_note = ""
    real = {k: v for k, v in diag.items() if "oracle" not in k}
    if real:
        key = headline if headline in real else list(real)[0]
        rep = real[key]
        for code, acc in sorted(rep["per_error_code"].items(), key=lambda x: x[1]):
            klass = "ok" if acc > 0.95 else ("mid" if acc > 0.5 else "no")
            pills += (f'<span class="pill">{code} '
                      f'<b class="{klass}">{acc*100:.0f}%</b></span>')
        accuracy_note = (
            f'<b>{key.replace("_", " ")}</b> — {rep["accuracy"]*100:.1f}% overall, '
            f'{rep["terminal_missed"]} terminal cases missed.')
        others = [(k, v) for k, v in real.items() if k != key]
        if others:
            k2, r2 = others[0]
            accuracy_note += (
                f' For comparison, <b>{k2.replace("_", " ")}</b> scores '
                f'{r2["accuracy"]*100:.1f}% with {r2["terminal_missed"]} missed.')

    by_txn = defaultdict(list)
    for e in audit:
        by_txn[e["txn_id"]].append(e)

    # Label each payment with what makes it worth looking at, and float the
    # interesting ones to the top. Scrolling 40 identical-looking options
    # hunting for the one double charge is not a reasonable thing to ask.
    def _label(steps):
        outs = [s.get("outcome") for s in steps]
        if "DOUBLE_CHARGE" in outs:
            return "DOUBLE CHARGE — customer had already paid", 0
        if any(s["decision"] == "BLOCKED" for s in steps):
            reason = next(s["reason"] for s in steps if s["decision"] == "BLOCKED")
            return f"BLOCKED — {reason}", 1
        if any(s["proposed_action"] == "GIVE_UP" for s in steps):
            return "GAVE UP — terminal diagnosis, never retried", 2
        if "SUCCESS" in outs:
            return f"recovered after {len(steps)} step(s)", 3
        return f"not recovered — {len(steps)} step(s)", 4

    ranked = []
    for t, v in by_txn.items():
        text, rank = _label(v)
        ranked.append((rank, t, text))
    ranked.sort(key=lambda r: (r[0], r[1]))
    opts = "".join(
        f'<option value="{t}">{t} — {text}</option>'
        for _, t, text in ranked[:60]
    )

    llm_note = ""
    if any(r["strategy"] == "agent_llm_learned" for r in rows):
        llm_rows = [r for r in rows if r["strategy"] in
                    ("agent_rules_learned", "agent_llm_learned")]
        if len(llm_rows) == 2 and llm_rows[0]["recovered_inr"] == llm_rows[1]["recovered_inr"]:
            llm_note = ('<div class="note alert"><b>The LLM arm is identical to the '
                        'rule arm.</b> Every transaction fell back to the rule table, '
                        'so this row carries no information. Check the console output '
                        'for the API error and re-run with <code>--strict</code>. '
                        'Do not report this row.</div>')

    html = f"""<!doctype html><meta charset="utf-8">
<title>Failed-payment recovery — results</title><style>{CSS}</style>
<div class="wrap">
<h1>Failed-payment recovery agent</h1>
<p class="sub">Held-out test split · {corpus['n']:,} failed payments ·
{_inr(corpus['at_risk_inr'])} at risk · the train split is never scored</p>

<div class="cards">{card_html}</div>

{llm_note}

<h2>Strategies compared on the same transactions</h2>
<table><tr><th>Strategy</th><th>Recovered</th><th>Rec. rate</th>
<th>% of ceiling</th><th>Actions/txn</th><th>Double charges</th>
<th>Wasted attempts</th></tr>{''.join(trs)}</table>
<div class="note"><b>How to read this.</b> Every row runs against the identical
set of transactions with the same random draws, so the differences are policy,
not luck. <i>% of ceiling</i> is measured against the recoverable subset —
risk-blocked customers and those who already paid elsewhere cannot be recovered
by anyone. <i>Wasted attempts</i> are debits fired at instruments that could not
have worked at that moment; the oracle row is 0 by construction.</div>

<h2>Diagnosis accuracy by error code</h2>
<div class="acc">{pills}</div>
<div class="note">{accuracy_note}<br><br>
Scored against the hidden root cause the diagnoser never sees. Higher accuracy
here does <i>not</i> mean more money recovered. The classifier is trained on
weak labels derived from retry outcomes, so classes that take the same action
collapse into one — it scores 0% on a code whose true class it confuses with an
action-equivalent one, and still recovers more. What matters operationally is
the terminal-case count: payments the agent should never have retried at all.</div>

<h2>Audit trail</h2>
<p class="sub">Pick a payment to see every action taken or blocked, and why. The most interesting cases — double charges, blocked actions, terminal give-ups — are listed first.</p>
<select id="pick">{opts}</select>
<div class="steps" id="steps"></div>
</div>
<script>
const DATA = {json.dumps(dict(by_txn))};
const inr = v => "₹" + Number(v).toLocaleString("en-IN");
function render(id) {{
  const el = document.getElementById("steps");
  el.innerHTML = (DATA[id] || []).map(s => `
    <div class="step ${{s.decision === "BLOCKED" ? "blocked" : ""}}">
      <div class="tag ${{s.decision}}">${{s.decision}}</div>
      <div>
        <div class="a">${{s.proposed_action}}
          <span style="color:#8b949e;font-weight:400">· ${{s.at.replace("T"," ").slice(0,16)}}</span></div>
        <div class="r">${{s.diagnosis}} (conf ${{(s.confidence*100).toFixed(0)}}%) — ${{s.reason}}</div>
        <div class="m">
          ${{s.expected_value_inr !== null ? "EV " + inr(s.expected_value_inr) + " · " : ""}}
          ${{s.outcome ? '<span class="' + s.outcome + '">' + s.outcome + "</span>" : ""}}
          ${{s.recovered_inr ? " · recovered " + inr(s.recovered_inr) : ""}}
        </div>
      </div>
    </div>`).join("");
}}
const pick = document.getElementById("pick");
pick.addEventListener("change", e => render(e.target.value));
render(pick.value);
</script>"""
    path.write_text(html, encoding="utf-8")