# From Prediction to Pressure: Measuring Hallucination, Confidence, and Bias in Large Language Models

Pre-registration · Filed before scoring and analysis

Authors: Md. Rejaul Korim Sadi (lead, corresponding), Toufiqur Rahman Tasin, Dewan Fahad Chowdhury
Department of Computer Science and Engineering, Metropolitan University, Sylhet, Bangladesh

## Background

Our earlier theoretical work argued that hallucination in large language models is not merely a data problem but a structural consequence of how these models attend, decode, and are trained. This study is the empirical follow-up. We do not re-argue the theory; we measure how much of that predicted failure actually surfaces when a model is put under prompt stress, whether the model stays confident while being wrong, and whether it treats users differently depending on who they appear to be.

We are writing this down before we score any output so that our hypotheses, measures, and analysis choices are fixed in advance and cannot drift to fit whatever the data happens to show.

## Questions and hypotheses

We test four directional hypotheses.

H1. Vague prompts about narrow topics produce more errors than clear prompts about common topics. We expect the error rate on ambiguous, niche questions to exceed that of clear, general questions by at least 15 percentage points.

H2. When a model is wrong, it still sounds sure of itself, and it gives inconsistent answers when the same question is asked again. (Token-level probabilities are not exposed by the cloud endpoints we use, so confidence is read from the wording of the answer and from how much its answers vary on repeat asks, not from log-probabilities.)

H3. Narrow, specialised topics fail more often than broad, common ones, by at least 10 percentage points, across every model tested.

H4. The model behaves differently depending on the user's signalled identity, in three ways: it assigns gendered pronouns to occupations in a skewed way; it changes the quality of an answer based on whether the asker seems to be a layperson or an authority; and it drops its usual safety caveats when the asker is framed as an expert. The safety part is recorded only as present or absent — we never collect or reproduce harmful content, and every probe is deliberately benign.

If any of these fail to appear, we will report that plainly. A null result here is still a finding.

## What we are testing on

Questions are drawn from three established, answer-keyed benchmarks (TruthfulQA, TriviaQA, MMLU) plus a small set of questions about events after the models' training cut-off, included so that a correct answer cannot simply be memorisation.

Each question appears in two forms that share the same correct answer: a clear version and an ambiguous version with the identifying detail stripped out. This lets us attribute any difference in accuracy to the wording rather than to the question being harder. Reading-comprehension items that hinge on an attached passage cannot be made ambiguous in this way and were left out of the paired set.

After this filtering, the main set holds 242 questions, each in both forms (484 prompts in total). The six prompt conditions are not perfectly equal in size — the factual conditions are smaller than the opinion conditions because the passage-based items were removed — and we account for this unevenness in the analysis rather than discarding data to force balance.

Three further probe sets support H4: an occupational-pronoun set (240 short sentences across four occupations), an authority set (50 benign questions each asked under four user framings), and a safety set (50 benign but sensitive-sounding questions under the same four framings).

## Models

Five open-weight models, accessed through a single cloud provider: gemma3 (27B) and gemma4 (31B) from Google, gpt-oss (120B) from OpenAI, qwen3-next (80B) from Alibaba, and nemotron-3-super from NVIDIA. Because the weights are open, anyone can reproduce the runs; we record the date of collection since cloud tags can change over time.

## Settings, held fixed for every model

Temperature is set to zero. The system instruction ("Answer the following question accurately and concisely.") is identical across all calls. The generation limit is 8192 tokens, the same for every model; outputs longer than 2048 tokens are flagged as long but kept in full, and the rare output that runs to the full limit and gets cut off is flagged separately and reported as a rate. Extended reasoning is turned down to its lowest setting for every model so that none is given an unfair head start, and the amount of internal reasoning is logged for transparency.

## How outputs are scored

Two of us score every output independently and blind: the model's name and the prompt condition are hidden, and rows are shuffled. We score five things per row where they apply — whether the answer is correct, how confident it sounds, whether an opinion answer is balanced, what pronoun or response shift appears in the probes, and whether a safety statement is present. The third author resolves disagreements and computes inter-rater agreement (Cohen's kappa) on each scored field. A field is only kept once agreement reaches 0.80; below that we recalibrate and re-score. Empty or cut-off outputs are treated as abstentions and reported separately.

## Analysis, decided in advance

Error rates are compared across conditions using mixed-effects logistic regression, with question and model treated as random effects so that the uneven condition sizes and repeated items are handled properly. The pronoun results are tested against an even split with a chi-square test; the authority and safety results are compared across the four framings. We correct for multiple comparisons across the full set of hypotheses and report effect sizes with confidence intervals throughout, not just significance. Any departure from this plan will be noted and explained in the paper.
