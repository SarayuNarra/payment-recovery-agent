# Results

Held-out test split: 2400 failed payments (seed 7), ₹8,324,252 at risk.

| Strategy | Recovered ₹ | Rec. rate | % of ceiling | Net ₹ | Cost/₹ rec. | Actions/txn | Double charges | Wasted attempts |
|---|---|---|---|---|---|---|---|---|
| do_nothing | 0 | 0.0% | 0.0% | 0 | — | 0.00 | 0 | 0 |
| immediate_retry_x2 | 2,331,056 | 28.0% | 32.3% | 2,322,660 | 0.0036 | 1.75 | 0 | 1,056 |
| fixed_24h_x3 | 3,925,386 | 47.2% | 54.4% | 3,903,008 | 0.0057 | 2.32 | 75 | 1,540 |
| link_blast | 4,706,408 | 56.5% | 65.3% | 4,697,399 | 0.0019 | 2.08 | 35 | 472 |
| agent_rules_handwritten | 5,230,013 | 62.8% | 72.5% | 5,219,638 | 0.0020 | 1.92 | 28 | 270 |
| agent_rules_learned | 5,911,887 | 71.0% | 82.0% | 5,901,268 | 0.0018 | 1.61 | 24 | 193 |
| agent_localclf_learned | 6,005,540 | 72.2% | 83.3% | 5,995,205 | 0.0017 | 1.50 | 25 | 0 |
| [upper bound] agent_oracle_learned | 6,059,345 | 72.8% | 84.0% | 6,048,914 | 0.0017 | 1.49 | 26 | 0 |
