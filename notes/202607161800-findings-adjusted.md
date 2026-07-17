# Findings — adjusted family (which baseline features accompany later gain)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)); reviewed and extended on 2026-07-17 to cover all models in the family. Preliminary.

## What these models ask

"Among children alike on everything else in the model, which **baseline** features go with **more later gain** in word reading?" Each predictor is reported twice — its **mutually-adjusted** coefficient (holding all the other baseline measures fixed) and its **bivariate** coefficient (on its own) — because those answer different questions and often disagree. That disagreement is the point: a predictor can look strongly linked to gain on its own yet fade to nothing once its correlated companions are held fixed (or vice versa). Reading the adjusted column as if it were a set of independent effects is the "Table-2 fallacy", and the two columns here are laid out side by side precisely so a reader can see where it would bite. Nothing in this family is causal: these are **between-child associations** describing _who_ progresses, not levers anyone could pull. The two models are the RLI trial cohort (LRP-RLI-ADJ-065, word reading measured by the EWRSWR item count) and a separate, non-randomised Byrne historical cohort (LRP-RLM-ADJ-001, #338). Both are Beta-Binomial ANCOVA-style models on a bounded post-score count, linear in every predictor — no interaction, threshold or knee terms are estimated.

A few Bayesian terms used below, in frequentist-friendly language. A **95% credible range** (or interval) is the range within which the parameter lies with 95% probability _given the data and priors_ — unlike a confidence interval, that is a direct probability statement about the parameter itself. **P(effect > 0)** (the "direction probability") is the posterior probability that the true association is positive; we grade evidence on whichever direction the probability favours, using the project evidence ladder (#179): `inconclusive` < 0.75 ≤ `suggestive` < 0.91 ≤ `moderate` < 0.97 ≤ `strong` < 0.99 ≤ `very strong`. That ladder grades our confidence in the _direction_ of the association, not that it is large enough to matter. These models do not ship a ROPE ("region of practical equivalence", a band around zero deemed too small to matter) table, so no "big enough to matter" verdict is reported for individual predictors; magnitudes are given on the natural items scale so a reader can judge practical size directly. Note that the EWRSWR outcome is a **capped word-reading item count**, so at the top of the scale a child cannot register further gain — part of any "older starters gained less" pattern reflects that ceiling and the measurement structure of a bounded count, not a behavioural fact.

## Convergence gate

Both models **passed** cleanly. LRP-RLI-ADJ-065: 0 divergences, max R-hat 1.0007, min ESS ≈ 9,600, per-chain BFMI 0.91–0.97. LRP-RLM-ADJ-001: 0 divergences, max R-hat 1.0004, min ESS ≈ 14,200, per-chain BFMI 0.96–1.02. No parameter is near a gate threshold in either fit.

## Results — all models

Magnitudes are on the natural **items** scale (difference in word-reading items associated with a 1-SD-higher baseline value of the predictor, holding the other predictors fixed); the "adjusted std" column gives the same coefficient on the standardised logit scale for comparison. "Dir. prob" is the posterior probability in the favoured direction, graded against the ladder above. **Causal status: association (adjusted, between-child, latent-ability-confounded) for every row — none is a randomised contrast.**

### LRP-RLI-ADJ-065 — RLI trial cohort, word reading (EWRSWR item count)

Adjusted (mutually-adjusted) associations, strongest direction-evidence first:

| Predictor                       | Adjusted median (items) | 95% credible range | Adjusted std (logit) | Dir. prob      | Evidence     |
| ------------------------------- | ----------------------- | ------------------ | -------------------- | -------------- | ------------ |
| Age (T1)                        | −2.9                    | −4.6 to −1.0       | −0.318               | 0.997 negative | very strong  |
| Hearing status (T1)             | +2.4                    | +0.17 to +4.87     | +0.215               | 0.978 positive | strong       |
| Speech-missing indicator        | −2.0                    | −4.34 to +0.38     | −0.218               | 0.944 negative | moderate     |
| Behaviour (T1)                  | −1.6                    | −3.54 to +0.35     | −0.174               | 0.941 negative | moderate     |
| Letter sounds (T1)              | +1.8                    | −0.85 to +4.76     | +0.158               | 0.888 positive | suggestive   |
| Language composite (T1)         | +1.8                    | −1.39 to +5.49     | +0.157               | 0.840 positive | suggestive   |
| Non-verbal MA (blocks, T1)      | +0.5                    | −1.78 to +3.03     | +0.044               | 0.647 positive | inconclusive |
| Hearing-missing indicator       | +0.4                    | −1.57 to +2.53     | +0.034               | 0.634 positive | inconclusive |
| Speech production (deapp_c, T1) | +0.6                    | −2.34 to +4.08     | +0.050               | 0.628 positive | inconclusive |
| Blending (T1)                   | +0.3                    | −1.68 to +2.49     | +0.024               | 0.590 positive | inconclusive |
| Phonological memory (erbto, T1) | −0.03                   | −2.84 to +3.19     | −0.011               | 0.526 negative | inconclusive |

Bivariate (each predictor on its own, standardised logit scale) and how it compares to the adjusted column — this is where the Table-2 divergences live:

| Predictor                                 | Bivariate std | Bivariate dir. prob | Bivariate evidence | vs adjusted                                                                     |
| ----------------------------------------- | ------------- | ------------------- | ------------------ | ------------------------------------------------------------------------------- |
| Letter sounds (T1)                        | +0.293        | 0.991 positive      | very strong        | **attenuates**: strong on its own, only suggestive once other skills held fixed |
| Behaviour (T1)                            | −0.250        | 0.987 negative      | strong             | attenuates to moderate when adjusted                                            |
| Language composite (T1)                   | +0.192        | 0.937 positive      | moderate           | attenuates to suggestive when adjusted                                          |
| Hearing status (T1)                       | +0.164        | 0.952 positive      | moderate           | **strengthens**: adjusted evidence (strong) exceeds bivariate                   |
| Age (T1)                                  | −0.196        | 0.976 negative      | strong             | strengthens to very strong when adjusted                                        |
| Non-verbal MA (blocks)                    | +0.111        | 0.857 positive      | suggestive         | fades to inconclusive when adjusted                                             |
| Speech production (deapp_c)               | −0.092        | 0.795 negative      | suggestive         | **flips sign**: suggestive-negative alone, inconclusive-positive adjusted       |
| Blending / hearing-missing / phon. memory | ≈ 0           | 0.51–0.69           | inconclusive       | inconclusive either way                                                         |

The headline (clearest adjusted predictor) matches `key_findings.json` exactly: a 1-SD-older child at T1 goes with **−2.9 items** of word-reading difference (95% range −4.6 to −1.0; 99.7% probability negative, very strong). The second well-resolved adjusted predictor is **hearing status** at **+2.4 items** (95% range +0.17 to +4.87; 97.8% positive, strong) — better baseline hearing accompanies more gain, and this signal is _sharper_ after adjustment than on its own, so it is not an artefact of correlation with the other baseline skills. Letter sounds and language look like clear positive predictors bivariately (very strong / moderate) but attenuate to merely suggestive once the correlated baseline skills are held fixed — a textbook Table-2 divergence. Behaviour and the speech-missing indicator carry moderate negative adjusted evidence.

**SES sensitivity check** (`ses_sensitivity.csv`, the n = 39 subset that also has mother's post-16 education). Adding SES leaves the age signal essentially intact (std −0.267, 97.6% negative, still strong) and actually _strengthens_ two predictors: language composite rises to std +0.328 (97.8% positive, strong) and hearing status to std +0.271 (99.0% positive, very strong). Mother's post-16 education itself enters at std −0.132 (87.7% negative, suggestive) — a mildly counter-intuitive negative that should be read cautiously given the small subset and is not something to over-interpret.

### LRP-RLM-ADJ-001 — Byrne historical cohort, word-reading-predictor gain (waves 1–3)

A _separate, non-randomised historical study_; every estimate here is descriptive or adjusted-associational by design. Items-scale magnitudes are on this study's own larger nominal scale and are **not** directly comparable to the RLI item counts (different measure, different SD), so the bigger numbers do not mean a bigger effect.

| Predictor                                    | Adjusted median (items) | 95% credible range | Adjusted std (logit) | Dir. prob      | Evidence     |
| -------------------------------------------- | ----------------------- | ------------------ | -------------------- | -------------- | ------------ |
| Age (months)                                 | −8.3                    | −13.3 to −3.3      | −0.377               | 0.999 negative | very strong  |
| BAS recall of digits (basdig)                | +4.6                    | −1.99 to +11.0     | +0.210               | 0.908 positive | suggestive   |
| BAS similarities / verbal reasoning (bassim) | +2.2                    | −4.14 to +8.57     | +0.099               | 0.744 positive | inconclusive |
| TROG receptive grammar (trog)                | −2.1                    | −8.25 to +4.08     | −0.093               | 0.737 negative | inconclusive |
| BAS number skills (basnum)                   | +0.3                    | −7.17 to +7.72     | +0.015               | 0.531 positive | inconclusive |
| BPVS receptive vocabulary (bpvs)             | +0.1                    | −5.21 to +5.54     | +0.005               | 0.512 positive | inconclusive |

Bivariate column here broadly tracks the adjusted one — age is very strong negative either way (bivariate std −0.374, 99.9% negative) and BAS recall of digits is the clearest non-age predictor in both (bivariate std +0.225, 93.1% positive, moderate; adjusted suggestive). The one notable divergence is BPVS receptive vocabulary, which is inconclusive-positive when adjusted but suggestive-_negative_ on its own (bivariate std −0.099, 79.4% negative) — a sign-flip of the same Table-2 flavour, though at inconclusive-to-suggestive strength it should not be leaned on. The headline matches `key_findings.json`: 1-SD-older (in months) accompanies **−8.3 items** of difference (95% range −13.3 to −3.3; 99.9% probability negative, very strong).

## The one-paragraph story

In both cohorts the single clearly-resolved predictor is **baseline age**: among children alike on the other baseline measures, older starters show _less_ measured gain over the window (RLI −2.9 items, very strong; Byrne −8.3 items, very strong). This describes the sample's progress pattern and partly reflects the ceiling and measurement structure of a bounded, capped item count — it is **not** a claim that being older causes less gain, and certainly not something the intervention changes. Beyond age, the RLI cohort has a second genuinely well-resolved adjusted association — **hearing status** (+2.4 items, strong positive), which sharpens rather than fades under adjustment — plus suggestive positive links for baseline letter sounds and language and moderate negative links for behaviour and a speech-missing flag. The Byrne cohort adds only a suggestive positive BAS recall-of-digits (+4.6 items); everything else there is inconclusive. Several predictors' adjusted and bivariate coefficients diverge (letter sounds, language, behaviour, hearing status, speech production, BPVS), which is exactly the Table-2 caution the design is built to surface. See the companion [ITT](202607161800-findings-itt.md) note for the randomised causal contrasts; this family carries none.

## What is causal

**Nothing.** These are adjusted between-child associations, confounded by latent general ability and everything else not in the model. The RLM model belongs to the Byrne descriptive study, where — with no randomisation — _every_ estimate is descriptive or adjusted-associational by design. Positive coefficients mean "higher baseline value accompanies more later gain in these children", never "raising this would produce more gain".
