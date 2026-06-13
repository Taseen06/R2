# Inter-Rater Reliability — Cohen's Kappa Report (Final)
**Study:** From Prediction to Pressure (R2) · **Raters:** Sadi & Tasin (independent, blind) · **Adjudicator:** Chowdhury
**Basis:** 600 dual-scored rows (120 per task type). Pre-registered gate: κ ≥ 0.80 per column.

| Column | n | Cohen's κ | Status |
|---|---|---|---|
| correctness | 120 | 0.853 | PASS — almost perfect |
| confidence_tone | 359 | 1.000 | PASS (after recalibration) |
| opinion_balance | 120 | 1.000 | PASS — perfect |
| bias_signal | 120 | 1.000 | PASS — perfect |
| safety_statement | 120 | 1.000 | PASS — perfect |

**Outcome:** All five columns meet the pre-registered κ ≥ 0.80 gate. Dataset reliability is locked.

**Recalibration note (disclosed):** On the first scoring pass, confidence_tone fell below the gate (κ = 0.498): the two raters applied a consistent but different plain-vs-assertive boundary. Per the pre-registered protocol, the raters recalibrated to the locked operational definition (assertive = explicit emphatic/committed wording only; a confident flat declarative is plain — matching the rubric's own example "The capital is Paris"), then re-scored confidence_tone only. Post-recalibration κ = 1.000. All other columns were unchanged.

**Scale (Landis & Koch 1977):** 0.81–1.00 almost perfect; 0.61–0.80 substantial; 0.41–0.60 moderate.
