# Failed-payment recovery agent

Track 03 — AI Revenue Recovery.

A payment fails. Something in the world caused it: the customer is broke until
payday, the issuer is down, the card is dead, or the customer walked away from
the OTP screen. Each needs a different intervention at a different time, and the
error code alone often doesn't tell you which one you're in.

This agent diagnoses the cause, picks a bounded intervention ladder, and stops.
It is evaluated on a held-out split against four strategies production systems
actually use today, plus an oracle upper bound.

**No dependencies, no API keys, no network.** `python run_eval.py` reproduces
every number below from the standard library.

## Results

Held-out test split: **2,400 failed payments, ₹83.2L at risk.** The 1,600-row
train split is used only to fit the timing model and the text classifier, and is
never scored.

| Strategy | Recovered ₹ | Rec. rate | % of ceiling | Actions/txn | Double charges | Wasted attempts |
|---|---|---|---|---|---|---|
| do nothing | 0 | 0.0% | 0.0% | 0.00 | 0 | 0 |
| immediate retry ×2 | 23,31,056 | 28.0% | 32.3% | 1.75 | 0 | 1,056 |
| fixed 24h retry ×3 | 39,25,386 | 47.2% | 54.4% | 2.32 | 75 | 1,540 |
| link blast | 47,06,408 | 56.5% | 65.3% | 2.08 | 35 | 472 |
| agent, hand-written ladder | 52,30,013 | 62.8% | 72.5% | 1.92 | 28 | 270 |
| agent, learned ladder | 59,11,887 | 71.0% | 82.0% | 1.61 | 24 | 193 |
| **agent, learned ladder + text classifier** | **60,05,540** | **72.2%** | **83.3%** | **1.50** | 25 | **0** |
| _[upper bound] perfect diagnosis_ | _60,59,345_ | _72.8%_ | _84.0%_ | _1.49_ | _26_ | _0_ |

**₹13.0L more recovered than the best naive strategy, using 28% fewer attempts
per payment.** Zero wasted debits against instruments that could never have
worked, down from 472. Within 0.6pp of the theoretical ceiling.

"% of ceiling" is measured against the recoverable subset — risk-blocked
customers and those who already paid elsewhere cannot be recovered by anyone, so
100% is the wrong denominator and reporting against it would flatter every row
equally.

```bash
python run_eval.py --n 4000 --seed 7   # ~20s, writes results/report.html
python -m pytest tests/ -q             # 7 invariant tests
```

Open `results/report.html` for the same numbers with a clickable audit trail.

## Architecture

```
failed payment
      │
      ▼
┌─────────────┐  error code + raw gateway message + customer history
│  DIAGNOSE   │  → 1 of 7 classes + confidence + suggested wait
└─────────────┘  rules for the 9 unambiguous codes,
      │          trained text classifier for the 3 that aren't
      ▼
┌─────────────┐  learned P(success | class, flow, action, delay)
│    PLAN     │  → (action, delay) ladder ranked by expected value
└─────────────┘  fitted on train-split retry outcomes
      │
      ▼
┌─────────────┐  action budget · charge cap · quiet hours · contact fatigue
│    GATE     │  · recovery window · negative-EV veto
└─────────────┘  nothing reaches the payment rail without passing all six
      │
      ▼
┌─────────────┐  every action, executed or blocked, with its reason
│   AUDIT     │  → results/audit_sample.json, results/report.html
└─────────────┘
```

Three interventions exist: retry the same rail, retry an alternate instrument,
send a payment link. Ordering matters and the agent learns it — for a dead card,
an alternate rail at +2h beats a link; for an abandoned OTP a silent retry is
nearly worthless because authentication needs the customer present.

## Where the gains came from

Each layer is a separate row in the table, so the contribution is measured
rather than asserted.

| Change | Gain |
|---|---|
| Diagnosis-driven ladder vs blasting links | +6.3pp |
| Learning retry timing instead of guessing | +8.3pp |
| Text classifier on the ambiguous codes | +1.2pp |
| _(remaining headroom to perfect diagnosis)_ | _+0.6pp_ |

The learned timing is the biggest single win, and it is the least glamorous.
Some of what it discovered:

```
USER_INTENT_DROP / checkout
  RETRY_SAME_RAIL   +2h: 92.5%   +24h: 44.1%   +72h: 7.3%

INSTRUMENT_DEAD / checkout
  RETRY_SAME_RAIL   +2h:  1.2%   ← retrying a dead card is worthless at any delay
  RETRY_ALT_RAIL    +2h: 60.1%   ← routing around it is not
```

Nobody wrote those numbers down. They fall out of 6,400 randomised probes on the
train split.

## The finding that surprised me

The text classifier **raises recovery from 71.0% to 72.2% while lowering diagnosis
accuracy from 83.9% to 80.1%.**

That looks like a contradiction and isn't. Its training labels are recovered from
retry outcomes — what worked — not from the true cause:

```
same-rail retry succeeds early        → nothing was broken
same-rail fails, alternate rail works → the instrument was the problem
early attempts fail, a late one works → it was a timing problem
nothing ever works, on any rail       → terminal; stop
```

A daily-limit failure and a dead card both look like "alternate rail works", so
they collapse into one label. The classifier therefore scores 0% on
`GATEWAY_TIMEOUT` — it calls those `USER_INTENT_DROP` when the truth is
`TRANSIENT_SHORT` — and yet recovers more money, because both classes take the
same action: retry early.

Meanwhile it fixes what actually mattered. `DO_NOT_HONOUR` goes from **0% to
66.6%**, and transactions where the correct call was "stop, never retry" go from
**74 missed to 0**.

The lesson I'd take to production: diagnosis accuracy against true cause is the
wrong yardstick for this problem. Action-equivalent classes are free to be
confused. Both numbers are reported here rather than only the flattering one.

## Why there is no LLM in the results table

There is an LLM diagnoser in `recovery/diagnoser.py`, with batching, dedup, disk
caching, and graceful fallback. It is not in the headline result, on purpose.

The oracle row bounds *any* diagnoser's remaining contribution at **+1.8pp** over
the rule table. The local naive Bayes classifier — 40 lines, no dependencies,
runs in 200ms — captures **1.2pp of that 1.8pp**, leaving at most 0.6pp for a
language model to win. That is not worth an API dependency, a per-transaction
inference cost, and a network failure mode in a retry pipeline that has to run
unattended.

Building the ceiling estimate *first* is what made that call possible. Without
the oracle row I would have shipped the LLM and reported a number with nothing to
compare it to.

If you want to run it anyway: `ANTHROPIC_API_KEY=... python run_eval.py --llm`.
It routes only the 3 ambiguous codes, dedups 915 rows down to 132 distinct
questions, and caches to `results/llm_cache.json`, so a full run is 6 API calls.

## The failure handled gracefully

About 6% of customers pay through another channel while a recovery sequence is in
flight. Charging after that isn't a neutral no-op — it's a double charge, a
refund, a support ticket, and a customer who doesn't come back. The simulator
models it at ₹150 of downstream cost and it is a scored column, not a footnote.

`fixed_24h_x3` causes 75. The agent causes 25, because it stops on success,
respects the action budget, and abandons terminal diagnoses instead of grinding
through a fixed schedule.

## Bounds

Enforced centrally in `Gatekeeper`, deliberately separate from the policy so they
can be audited on their own:

- max 4 actions per transaction, of which max 3 are charge attempts
- max 2 customer contacts per 72 hours, min 24 hours apart
- no customer contact between 21:00 and 08:00
- 16-day recovery window, then stop
- never retry a `RISK_TERMINAL` diagnosis — a policy decision, hard-coded rather
  than left to a learned model
- negative expected value vetoes the action, so a ₹49 failure isn't chased three
  times

Blocked actions are written to the audit trail with the reason, which is what
makes the bound checkable rather than claimed. `tests/test_core.py` asserts all
of them hold across thousands of generated transactions.

## Why the evaluation isn't circular

Each transaction is generated from a **hidden root cause** plus hidden state:
when funds arrive, when an outage ends, whether the instrument is permanently
dead, whether the customer will pay elsewhere. The agent reads none of it. It
sees the error code, the gateway message, the rail, the flow, the amount, and
90-day history — the fields a real recovery system has.

The simulator is the only component that can read hidden state, and it resolves
every attempt against it. Its RNG is keyed on `(txn_id, attempt_number)`, so two
strategies making the same call at the same moment get the same draw; the
comparison is like-for-like.

The text classifier's training labels come from probe *outcomes*, never from
hidden state, which is why they are noisy and outcome-shaped rather than clean.

## Known limitations

- **The corpus is synthetic.** Structure is drawn from public knowledge of Indian
  payment failure modes, not Razorpay data. The generator's message templates make
  the ambiguous codes cleanly separable by a careful reader; real gateway messages
  are noisier, so the classifier's real-world edge is probably smaller than shown.
- **Weak labels assume you can probe.** The train split is probed with a fixed
  battery of six retries. Real logs only contain retries someone chose to make, so
  they carry survivorship bias this doesn't reproduce.
- **Intent decay is an assumption** — an exponential with a per-transaction
  half-life. The shape is plausible; the parameters are calibrated against nothing.
- **No live Razorpay integration.** Everything runs against the simulator. The
  action interface is narrow (`retry`, `retry_alt`, `send_link`), so swapping in
  test-mode Payments and Payment Links is a small change, but it isn't done here.
- **Single-transaction scope.** It doesn't reason about a customer with several
  failing subscriptions at once, where the right move is one conversation rather
  than three sequences.

## Layout

```
recovery/corpus.py      synthetic corpus + hidden ground truth
recovery/simulator.py   outcome model; the only reader of hidden state
recovery/diagnoser.py   rule / LLM / hybrid / oracle diagnosers + accuracy
recovery/textclf.py     weak labels from retry outcomes + naive Bayes
recovery/learn.py       timing model fitted on train-split probes
recovery/agent.py       policy ladder, gatekeeper, audit trail
recovery/evaluate.py    baseline strategies + metric aggregation
recovery/report.py      static HTML report generator
run_eval.py             train/test split, benchmark, writes results/
tests/test_core.py      invariants the results table depends on
```
