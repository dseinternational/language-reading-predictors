> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# Full reporting fit of all statistical (Bayesian) models — findings

Date: 2026-07-08. A clean full fit of the entire Bayesian statistical-model suite from `main` at commit `8ec2089`, under the production sampling configuration, with every Quarto report rendered and all artefacts uploaded to Azure Blob Storage. This note records what was run, how well it converged, and what the fits say — read alongside each model's own report and `METHODS.md`.

## What was run

`python scripts/fit_statistical_model.py all --config reporting --render`, then `scripts/compare_statistical_models.py --config reporting`, then upload.

- **89/89 models fitted, 0 failed. 89/89 Quarto reports rendered.** Wall-clock **1 h 44 m** on a 16-core VM.
- Sampling preset `reporting`: **6000 draws × 6000 tune × 6 chains, `target_accept = 0.95`** per model (36 000 posterior draws each).
- The 16 model families, by count: itt 23, gain_factors 16, aligned 9, did 8, level_factors 8, mechanism 8, dose_response 3, joint 3, growth 2, horseshoe 2, mediation 2, plus one each of lcsm, mediation_multi, corr_factor and historical_growth (the last a separate historical-cohort study).
- The Quarto renders emit non-fatal "problem with a fenced div" warnings (a callout-syntax quirk); all 89 `index.html` built regardless.

## Artefacts and upload

Uploaded to Azure Blob Storage under `<outputs container>/language-reading-predictors/output/statistical_models/`:

- `models/{model_id}-reporting/` — **7228 files, 479 MB, 0 failed** (config, diagnostics, priors, result CSVs, PNG/SVG figures, and the rendered `index.html`).
- `comparison/` — 9 files (τ forests, ITT-vs-joint consistency, mechanism/dose/DiD nested-LOO tables).

Trace files (`trace.nc`, ~16.8 GB across the 89 fits) were **excluded** from the upload by design, matching the fit script's default. Note that the built-in `--upload` flag could not be used: it authenticates through the Azure Python SDK against the `public` container, for which the VM's managed identity has no write permission (`AuthorizationPermissionMismatch`). The upload was done instead with `azcopy` (managed identity) to the `outputs` container, the same path the gradient-boosting artefacts use. Output is git-ignored, so nothing beyond this note is committed.

## Convergence and diagnostics

Every model was checked against the pass/fail gate (`r̂ ≤ 1.01`, ESS ≥ 400, BFMI ≥ 0.30, and zero divergences). **79/89 passed outright.** All 10 that did not have **`r̂ = 1.00` and pass the ESS check** — they are flagged purely on divergences, not on non-convergence:

| Model            | Family                     | Divergences | of 36 000 | Other flags          |
| ---------------- | -------------------------- | ----------: | --------: | -------------------- |
| lrp-rli-mm-001   | corr_factor                |         422 |    1.17 % | **BFMI 0.21 < 0.30** |
| lrp-rli-mech-056 | mechanism (R→W)            |         145 |    0.40 % | —                    |
| lrp-rli-mech-058 | mechanism (L→W)            |          43 |    0.12 % | —                    |
| lrp-rli-mech-057 | mechanism (E→W)            |          39 |    0.11 % | —                    |
| lrp-rli-mech-071 | mechanism (L→W, moderated) |          34 |    0.09 % | —                    |
| lrp-rli-mech-073 | mechanism (L→W × age)      |          30 |    0.08 % | —                    |
| lrp-rli-mech-173 | mechanism (L→W baseline)   |          28 |    0.08 % | —                    |
| lrp-rli-dose-077 | dose_response              |          13 |    0.04 % | —                    |
| lrp-rli-did-007  | did (period dose)          |          10 |    0.03 % | —                    |
| lrp-rli-dose-177 | dose_response              |          10 |    0.03 % | —                    |

Nine of the ten sit at **≤ 0.4 % divergences** — a handful of transitions from the funnel geometry of the GP/HSGP mechanism surfaces and the session-dose slopes, comfortably inside the ≤ 1 % guidance in `METHODS.md`. Their τ/slope posteriors are trustworthy; the divergences are worth a note, not a veto. **Only `lrp-rli-mm-001` (the correlated-domain-factor measurement model) is a genuine sampling concern**: 1.17 % divergences _and_ sub-threshold BFMI (0.21), the signature of a latent-factor funnel that the current parameterisation samples inefficiently. Its factor **correlations** are stable and interpretable (below), but its **structural** coefficients should be read cautiously pending a non-centred reparameterisation or a higher `target_accept`.

## Headline causal findings — the randomised ITT effect (`itt` suite)

Only the on-intervention term is causal (the trial's random assignment identifies it with an empty adjustment set); every other coefficient in these models is an adjusted association. τ is reported below on the **risk-difference / probability scale** (median, 95 % credible interval, and posterior probability the effect is positive). Positive = the intervention helps.

| Outcome                        | Model   |     τ (RD) | 95 % CrI         | P(τ>0) | Read as                                    |
| ------------------------------ | ------- | ---------: | ---------------- | -----: | ------------------------------------------ |
| Letter-sound knowledge (L)     | itt-007 | **+0.110** | [+0.040, +0.179] |  0.999 | credible, large                            |
| Phoneme blending (B)           | itt-008 | **+0.099** | [+0.004, +0.192] |  0.980 | credible                                   |
| Taught expressive vocab (TE)   | itt-002 | **+0.064** | [+0.006, +0.122] |  0.985 | credible                                   |
| Word reading (W)               | itt-010 | **+0.030** | [+0.004, +0.057] |  0.986 | credible, small                            |
| Taught receptive vocab (TR)    | itt-001 |     +0.057 | [−0.003, +0.117] |  0.968 | leans positive                             |
| Untaught receptive vocab (UR)  | itt-003 |     +0.050 | [−0.014, +0.116] |  0.937 | leans positive                             |
| Untaught expressive vocab (UE) | itt-004 |     +0.026 | [−0.041, +0.093] |  0.773 | inconclusive                               |
| Receptive vocab, ROWPVT (R)    | itt-005 |     +0.001 | [−0.027, +0.030] |  0.539 | null                                       |
| Expressive vocab, EOWPVT (E)   | itt-006 |     +0.001 | [−0.022, +0.025] |  0.534 | null                                       |
| Phonetic spelling (P)          | itt-009 |          — | —                |      — | floored; off-floor RD ≈ 0 (0.357 vs 0.360) |
| Nonword reading (N)            | itt-011 |          — | —                |      — | floored; inconclusive                      |

**The coherent story: the intervention credibly moves the reading/phonics-proximal and directly-taught skills — letter-sound knowledge, phoneme blending, word reading, and taught expressive vocabulary — while broad standardised vocabulary (ROWPVT, EOWPVT) is flat.** The gradient runs exactly as a phonics-and-taught-words intervention predicts: strongest on letter sounds, present on blending and word reading, present for _taught_ words but fading to inconclusive for _untaught_ words (the generalisation contrast itt-015 puts P(taught-expressive > untaught-expressive) at 0.79 — suggestive, not decisive), and absent on the broad norm-referenced vocabulary tests. The two heavily-floored outcomes (spelling, nonword reading) carry too little off-floor movement to estimate — for spelling the off-floor rate is essentially identical between arms (35.7 % vs 36.0 %).

## The effect is robust across adjustment and across three independent estimators

The word-reading and letter-sound effects survive both robustness adjustments and reappear, at consistent magnitude, in two designs that do not share the ITT model's assumptions:

- **General-ability adjustment** (block-design covariate, itt-017–024): L +0.110, W +0.028, TE +0.061 — essentially unchanged.
- **SES adjustment + matched complete-case comparators** (itt-013/113/014/114): L +0.107 to +0.123, W +0.030 — unchanged (the SES-adjusted W in itt-013 widens just across 0, but its complete-case sibling itt-014 stays credible).
- **Within-person waitlist-crossover DiD** (each child as its own control, `did` family): W δ +0.367 [+0.050, +0.685], L δ +0.560 [+0.183, +0.934], B δ +0.437 [+0.012, +0.851], TE +0.300 (P 0.97), R null — the same ranking, on the logit scale, from a completely different identification strategy.
- **DAG-faithful gain-factor ANCOVA** (`gf` family, on-intervention marginal): W +0.037 [+0.006, +0.065], L +0.097 [+0.029, +0.168], B +0.072 (P 0.91), F +0.049 (P 0.86), R/E null — agreeing a third time.
- **Levels view** (`lf` family, the only clean randomised term being the t2 group contrast): L +0.491 [+0.009, +0.963] credible; W/B/F lean positive but with wide single-timepoint intervals; R/E/T null.

Three estimators (ITT ANCOVA, within-person DiD, gain-factor ANCOVA) built on different assumptions converge on the **same causal ordering: letter sounds ≫ blending ≈ word reading > taught vocabulary ≫ broad vocabulary ≈ 0.**

## Mechanism and mediation — how the reading gain arises

These couplings are **adjusted associations, not causal drivers**, but they are internally consistent and answer the "through what?" question the trial itself cannot.

- **Mediation (g-formula NDE/NIE decomposition):** the intervention's effect on word reading runs **through letter-sound knowledge**. Single-mediator (med-059): NIE via L +0.023 [+0.006, +0.045], P 0.998; ≈ 62 % of the total effect mediated, with a non-credible direct effect. Two-mediator (med-064): NIE via L +0.025 [+0.006, +0.048] P 0.998, whereas NIE via expressive vocabulary is ≈ 0 [−0.009, +0.007]. The reading gain is carried by the code route, not the vocabulary route — the same conclusion the ITT ranking implies.
- **Mechanism GP slopes** (marginal, adjusted): E→W +0.271 [−0.001, +0.593] (the strongest, all but excluding 0), R→W +0.131, L→W +0.090 — all positive-leaning associations with word reading, credible intervals touching 0.
- **Latent coupled change-score model (lcsm-067):** prior letter-sound → later reading change +0.135 [+0.025, +0.256] P 0.99, and prior expressive-vocab → later reading change +0.284 [+0.054, +0.521] P 0.99 — both credible cross-lagged couplings; reading shows negative self-feedback (regression to the mean).
- **Dose-response:** pooled cumulative-session slope +0.127 [+0.028, +0.227] P 0.99 (dose-277); period-resolved slopes ≈ +0.13 per period (P ≈ 0.95). More sessions track more word-reading gain (dose is a partial collider, so this is a sensitivity view, not a clean causal dose curve).

## Secondary and cross-check families

- **Regularised-horseshoe predictor ranking (cross-check of the gradient-boosting ranking):** for word-reading _level_ (hs-002), letter sounds (P 0.99) and expressive vocabulary (P 0.99) are selected decisively, then grammar and age. For word-reading _gain_ (hs-001) nothing is selected (top probability 0.59, age) — echoing the gradient-boosting result that change scores are near-noise. The Bayesian and gradient-boosting rankings agree.
- **Correlated-domain-factor measurement model (mm-001):** the three latent domains are strongly correlated — vocabulary↔grammar 0.80, vocabulary↔code 0.74, code↔grammar 0.65 (all credible) — supporting a correlated-skill-system reading of the battery. (Structural coefficients held back pending the reparameterisation noted above.)
- **Multivariate growth curves (gc-069/070):** between-child gamma associations are inconclusive for R/E/W and credibly positive only for grammar (T ≈ +0.11).
- **Aligned per-protocol (`al` family):** cohort contrast +0.217 [−0.109, +0.544], correctly flagged **non-causal** (confounded by age-at-onset and cohort timing) — no term in this family is presented as causal, by design.
- **Adjusted between-child association (adj-065):** of the baseline predictors of word-reading gain, only age is credible (−0.259, older children gaining less); language composite (+0.225, P 0.93) and letter sounds (+0.159, P 0.89) lean positive but are not decisive.
- **Historical-cohort growth reproduction (rlm-hg-001, separate study):** converged cleanly (r̂ 1.00, 0 divergences).

## Caveats

- **This is preliminary, small-sample research** (≈ 159 children after cleaning; some outcomes far fewer). Credible intervals are wide, and the "credible" calls above rest on 95 % intervals that in several cases only just exclude zero.
- **Only τ (and the DiD δ, and the gain-factor on-intervention marginal) is causal.** Every mechanism slope, mediation path, coupling, growth gamma, horseshoe coefficient and adjusted association is an adjusted association and must not be read as "X drives Y".
- **`lrp-rli-mm-001` needs a sampling fix** before its structural coefficients are quoted; its correlations are fine. The nine other divergence-flagged fits are usable as-is but the mechanism GP models would benefit from a re-run at a higher `target_accept` if any single slope becomes load-bearing.
