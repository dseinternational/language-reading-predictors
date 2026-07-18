# Findings — adjusted family (which baseline features accompany later gain)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers are read from the `reporting`-config refit under the median + inner-50% + outer-89% equal-tailed credible-interval standard (2026-07-18; see [the credible-interval standard note](202607172359-credible-interval-standard.md) and [the refit process note](202607161130-full-statistical-refit.md)); reviewed and extended on 2026-07-17 to cover all models in the family. Only the credible-interval brackets changed when we moved from 95% to 89% — medians, direction probabilities and evidence labels are unchanged. Preliminary.

## What these models ask

"Among children alike on everything else in the model, which **baseline** features go with **more later gain** in word reading?" Each predictor is reported **twice** — once **bivariately** and once **mutually-adjusted** — because those are two genuinely different questions, and the gap between the answers is itself informative.

**Bivariate (unadjusted, one predictor at a time).** Fit later gain on a single baseline measure, with nothing else in the model. The coefficient is the _total_ association between that measure and later gain: it bundles the predictor's own contribution together with any signal it merely _shares_ with other, correlated baseline features. The question it answers is marginal — "if the only thing I knew about a child were this one score, how well would it track their later gain?" This is close to what a simple correlation or a one-predictor regression reports.

**Mutually-adjusted (multivariable, all predictors at once).** Put every baseline predictor into a _single_ regression. Each coefficient is now the association of that predictor with gain _among children who are alike on all the other predictors in the model_ — i.e. the part of its signal that its companions do **not** already account for. The question it answers is incremental — "over and above the other baseline measures, does this one add anything?"

**Why the two diverge: the predictors are correlated.** Baseline skills in this cohort travel together — a child strong on letter sounds tends also to be strong on language, vocabulary and blending, partly because all of them load on the same underlying general ability. A bivariate coefficient cannot tell a predictor's _own_ signal apart from the signal it borrows from these correlated companions; adjustment is precisely the operation that separates the two by holding the companions fixed. Two things can then happen:

- **Attenuation (shrink towards zero).** A measure that looked strongly linked to gain on its own weakens once its companions are held fixed — it was largely standing in for them. Here **letter sounds** is the textbook case (very strong bivariately, only suggestive adjusted): most of its raw association was shared with the other baseline skills.
- **Strengthening or sign-flip (suppression).** Occasionally adjustment makes an association _larger_, or reverses its sign, because a correlated companion had been masking it. **Hearing status** sharpens under adjustment (so its signal is not an artefact of correlation with the other skills), and **speech production** flips from suggestive-negative on its own to inconclusive-positive adjusted — classic suppression patterns.

**The Table-2 fallacy — why we print both columns.** Each adjusted coefficient is conditional on its _own_ particular set of held-fixed companions, so the adjusted column is **not** a menu of independent effects that can all be read off and believed simultaneously, and it is certainly not a set of causal effects. Treating every row of one multivariable table as a clean, independent (let alone causal) effect is the "Table 2 fallacy" (Westreich & Greenland 2013, _Am J Epidemiol_ 177(4):292–298, [doi:10.1093/aje/kws412](https://doi.org/10.1093/aje/kws412)). Laying the bivariate and adjusted columns side by side is the honest antidote: it lets a reader _see_ where a predictor's apparent importance is its own and where it is borrowed.

Nothing in this family is causal either way: both columns are **between-child associations** describing _who_ progresses, confounded by latent general ability and everything else not in the model — not levers anyone could pull. The two models are the RLI trial cohort (LRP-RLI-ADJ-065, word reading measured by the EWRSWR item count) and a separate, non-randomised Byrne historical cohort (LRP-RLM-ADJ-001, #338). Both are Beta-Binomial ANCOVA-style models on a bounded post-score count, linear in every predictor — no interaction, threshold or knee terms are estimated.

A few Bayesian terms used below, in frequentist-friendly language. An **89% credible range** (or interval) is the range within which the parameter lies with 89% probability _given the data and priors_ — unlike a confidence interval, that is a direct probability statement about the parameter itself. **P(effect > 0)** (the "direction probability") is the posterior probability that the true association is positive; we grade evidence on whichever direction the probability favours, using the project evidence ladder (#179): `inconclusive` < 0.75 ≤ `suggestive` < 0.91 ≤ `moderate` < 0.97 ≤ `strong` < 0.99 ≤ `very strong`. That ladder grades our confidence in the _direction_ of the association, not that it is large enough to matter. These models do not ship a ROPE ("region of practical equivalence", a band around zero deemed too small to matter) table, so no "big enough to matter" verdict is reported for individual predictors; magnitudes are given on the natural items scale so a reader can judge practical size directly. Note that the EWRSWR outcome is a **capped word-reading item count**, so at the top of the scale a child cannot register further gain — part of any "older starters gained less" pattern reflects that ceiling and the measurement structure of a bounded count, not a behavioural fact.

## Convergence gate

Both models **passed** cleanly. LRP-RLI-ADJ-065: 0 divergences, max R-hat 1.0007, min ESS ≈ 9,600, per-chain BFMI 0.91–0.97. LRP-RLM-ADJ-001: 0 divergences, max R-hat 1.0004, min ESS ≈ 14,200, per-chain BFMI 0.96–1.02. No parameter is near a gate threshold in either fit.

## Results — all models

Magnitudes are on the natural **items** scale (difference in word-reading items associated with a 1-SD-higher baseline value of the predictor, holding the other predictors fixed); the "adjusted std" column gives the same coefficient on the standardised logit scale for comparison. "Dir. prob" is the posterior probability in the favoured direction, graded against the ladder above. **Causal status: association (adjusted, between-child, latent-ability-confounded) for every row — none is a randomised contrast.**

**Reading the covariate rows (mind the coding).** Three rows are 0/1 flags, not graded skills, and their sign is easy to misread. **Hearing status** (`hs`) = 1 marks a child with **impaired hearing _or_ a history of repeated ear infections** (0 = clear, the reference) — a _risk flag_, so a **positive** coefficient means that flag accompanies **more** gain, not "better hearing → more gain". The other two are **missing-data indicators**, added so that children with an unrecorded baseline score are not dropped from the model; each describes _who was fully assessed_, not a skill level:

- **Hearing-missing** (`hs_missing`) = 1 when baseline hearing status was **unknown**. Those children are set to the clear reference (`hs = 0`), and this indicator gives the unknown-hearing group its own intercept so they are not silently treated as genuinely clear. Its coefficient is the gain difference of the unknown-hearing group versus the recorded-clear group (here +0.4 items, inconclusive — the unknown group is not distinguishable from clear).
- **Speech-missing** (`deapp_c_missing`) = 1 when the baseline DEAP speech-production score was **unrecorded**; the value is mean-filled and this indicator absorbs the offset. It is distinct from the speech-production _level_ row (`deapp_c`): the level says how a child scored, the indicator says only whether they were scored.

### LRP-RLI-ADJ-065 — RLI trial cohort, word reading (EWRSWR item count)

Adjusted (mutually-adjusted) associations, strongest direction-evidence first:

| Predictor                       | Adjusted median (items) | 89% credible range | Adjusted std (logit) | Dir. prob      | Evidence     |
| ------------------------------- | ----------------------- | ------------------ | -------------------- | -------------- | ------------ |
| Age (T1)                        | −2.9                    | −4.3 to −1.3       | −0.318               | 0.997 negative | very strong  |
| Hearing status (T1)             | +2.4                    | +0.48 to +4.48     | +0.215               | 0.978 positive | strong       |
| Speech-missing indicator        | −2.0                    | −4.00 to +0.01     | −0.218               | 0.944 negative | moderate     |
| Behaviour (T1)                  | −1.6                    | −3.26 to +0.04     | −0.174               | 0.941 negative | moderate     |
| Letter sounds (T1)              | +1.8                    | −0.50 to +4.25     | +0.158               | 0.888 positive | suggestive   |
| Language composite (T1)         | +1.8                    | −0.96 to +4.86     | +0.157               | 0.840 positive | suggestive   |
| Non-verbal MA (blocks, T1)      | +0.5                    | −1.45 to +2.61     | +0.044               | 0.647 positive | inconclusive |
| Hearing-missing indicator       | +0.4                    | −1.29 to +2.18     | +0.034               | 0.634 positive | inconclusive |
| Speech production (deapp_c, T1) | +0.6                    | −1.94 to +3.47     | +0.050               | 0.628 positive | inconclusive |
| Blending (T1)                   | +0.3                    | −1.40 to +2.13     | +0.024               | 0.590 positive | inconclusive |
| Phonological memory (erbto, T1) | −0.03                   | −2.42 to +2.65     | −0.011               | 0.526 negative | inconclusive |

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

The headline (clearest adjusted predictor) matches `key_findings.json` exactly: a 1-SD-older child at T1 goes with **−2.9 items** of word-reading difference (89% range −4.3 to −1.3; 99.7% probability negative, very strong). The second well-resolved adjusted predictor is **hearing status** at **+2.4 items** (89% range +0.48 to +4.48; 97.8% positive, strong). Mind the coding (see above): `hs` = 1 is the **impaired-hearing / repeated-ear-infection** flag, so the positive sign means that flag accompanies **more** later measured gain, not less — a counter-intuitive association that is _sharper_ after adjustment than on its own, so it is at least not a mere proxy for the correlated baseline skills. Read it cautiously all the same: n ≈ 51, a coarse 0/1 flag pooling two rather different conditions, and the usual latent-ability confounding — plausible explanations range from the ear-infection component being transient, through flagged children receiving extra monitoring or support, to small-sample noise. It is emphatically **not** evidence that hearing difficulty aids reading. Letter sounds and language look like clear positive predictors bivariately (very strong / moderate) but attenuate to merely suggestive once the correlated baseline skills are held fixed — a textbook Table-2 divergence. Behaviour and the speech-missing indicator carry moderate negative adjusted evidence.

**SES sensitivity check** (`ses_sensitivity.csv`, the n = 39 subset that also has mother's post-16 education). Adding SES leaves the age signal essentially intact (std −0.267, 97.6% negative, still strong) and actually _strengthens_ two predictors: language composite rises to std +0.328 (97.8% positive, strong) and hearing status to std +0.271 (99.0% positive, very strong). Mother's post-16 education itself enters at std −0.132 (87.7% negative, suggestive) — a mildly counter-intuitive negative that should be read cautiously given the small subset and is not something to over-interpret.

### LRP-RLM-ADJ-001 — Byrne historical cohort, word-reading-predictor gain (waves 1–3)

A _separate, non-randomised historical study_; every estimate here is descriptive or adjusted-associational by design. Items-scale magnitudes are on this study's own larger nominal scale and are **not** directly comparable to the RLI item counts (different measure, different SD), so the bigger numbers do not mean a bigger effect.

| Predictor                                    | Adjusted median (items) | 89% credible range | Adjusted std (logit) | Dir. prob      | Evidence     |
| -------------------------------------------- | ----------------------- | ------------------ | -------------------- | -------------- | ------------ |
| Age (months)                                 | −8.3                    | −12.6 to −4.0      | −0.377               | 0.999 negative | very strong  |
| BAS recall of digits (basdig)                | +4.6                    | −0.94 to +10.1     | +0.210               | 0.908 positive | suggestive   |
| BAS similarities / verbal reasoning (bassim) | +2.2                    | −3.20 to +7.56     | +0.099               | 0.744 positive | inconclusive |
| TROG receptive grammar (trog)                | −2.1                    | −7.31 to +3.20     | −0.093               | 0.737 negative | inconclusive |
| BAS number skills (basnum)                   | +0.3                    | −6.09 to +6.60     | +0.015               | 0.531 positive | inconclusive |
| BPVS receptive vocabulary (bpvs)             | +0.1                    | −4.43 to +4.74     | +0.005               | 0.512 positive | inconclusive |

Bivariate column here broadly tracks the adjusted one — age is very strong negative either way (bivariate std −0.374, 99.9% negative) and BAS recall of digits is the clearest non-age predictor in both (bivariate std +0.225, 93.1% positive, moderate; adjusted suggestive). The one notable divergence is BPVS receptive vocabulary, which is inconclusive-positive when adjusted but suggestive-_negative_ on its own (bivariate std −0.099, 79.4% negative) — a sign-flip of the same Table-2 flavour, though at inconclusive-to-suggestive strength it should not be leaned on. The headline matches `key_findings.json`: 1-SD-older (in months) accompanies **−8.3 items** of difference (89% range −12.6 to −4.0; 99.9% probability negative, very strong).

## The one-paragraph story

In both cohorts the single clearly-resolved predictor is **baseline age**: among children alike on the other baseline measures, older starters show _less_ measured gain over the window (RLI −2.9 items, very strong; Byrne −8.3 items, very strong). This describes the sample's progress pattern and partly reflects the ceiling and measurement structure of a bounded, capped item count — it is **not** a claim that being older causes less gain, and certainly not something the intervention changes. Beyond age, the RLI cohort has a second genuinely well-resolved adjusted association — a baseline **impaired-hearing / repeated-ear-infection flag** (`hs` = 1) at **+2.4 items, strong positive**, which sharpens rather than fades under adjustment. The positive sign is counter-intuitive (the flag accompanies _more_ measured gain, not less) and should be read cautiously rather than as "hearing difficulty helps" — see the results section. There are also suggestive positive links for baseline letter sounds and language and moderate negative links for behaviour and a speech-missing (data-availability) flag. The Byrne cohort adds only a suggestive positive BAS recall-of-digits (+4.6 items); everything else there is inconclusive. Several predictors' adjusted and bivariate coefficients diverge (letter sounds, language, behaviour, hearing status, speech production, BPVS), which is exactly the Table-2 caution the design is built to surface. See the companion [ITT](202607161800-findings-itt.md) note for the randomised causal contrasts; this family carries none.

## Latent general ability vs the block-design predictor

This note repeatedly calls every association "latent-ability-confounded" _and_ lists "Non-verbal MA (blocks)" as one of the predictors — which can look contradictory. It is not, because the two are different objects:

- **General ability (`GA`)** is the study DAG's **latent** node: an _unobserved_ common cause with an arrow into essentially every skill and into the reading outcome. Because it is never measured, no set of observed covariates can fully close the backdoor path `predictor ← GA → reading` sitting behind these associations — that irreducible residual is exactly what "latent-ability-confounded" names, and it is why only the randomised arm contrast (in the [ITT](202607161800-findings-itt.md) note, not here) escapes it.
- **Baseline non-verbal ability (`blocks`)** is an _observed_ t1 test — the WPPSI-III Block Design subtest. It is a single, noisy **indicator** of _one slice_ (non-verbal reasoning) of that broader latent construct — not the construct itself, and not a DAG node.

So putting block design in the model **shrinks** the general-ability confounding but never removes it: it captures only the non-verbal part of general ability, and with measurement error, leaving verbal ability, memory and the test's own noise still doing confounding work. That is why block design's _own_ coefficient (+0.5 items, inconclusive) is still an association, not "the causal effect of non-verbal ability" — and why every row here, block design included, keeps the latent-ability caveat. Conditioning on it (as this model and the ITT robustness models do) is a useful, pre-treatment **partial proxy** for `GA`, not a substitute for observing it. In the adjustment-set machinery this is explicit: `GA` is treated as _hypothetically_ blockable to define the best _observable_ set precisely because in reality it is not, so the models close every observable backdoor and carry the `GA` residual openly.

## What is causal

**Nothing.** These are adjusted between-child associations, confounded by latent general ability and everything else not in the model. The RLM model belongs to the Byrne descriptive study, where — with no randomisation — _every_ estimate is descriptive or adjusted-associational by design. Positive coefficients mean "higher baseline value accompanies more later gain in these children", never "raising this would produce more gain".
