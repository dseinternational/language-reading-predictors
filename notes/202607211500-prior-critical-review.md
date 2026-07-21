> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

# Critical prior-analysis review: language-reading-predictors Bayesian model suite

## Executive summary

**Headline judgement: the priors are fit for the study's interpretive goals, subject to a set of bounded caveats and one systemic diagnostic gap.** Across all 22 families and 179 fits the priors are genuinely weakly-informative on the logit scale, correctly centred, and correctly scaled to the study's small-sample regime (n≈50–60 for RLI, n≈54–97 for RLM). Prior-predictive coverage is sound everywhere it was computed, and where the data are informative they dominate the prior. No reported estimand is manufactured by its prior; the failure mode that does appear is the benign one, a sceptical zero-centred prior mildly _attenuating_ a real effect, not inventing one.

The verdict is nonetheless **"adequate with minor concerns" rather than "clean"**, for three reasons:

1. **Diagnostic coverage is uneven.** Power-scaling (psense) was run for only six of the 22 families (`bx`, `did`, `gf`, `itt`, `lcf`, `lf`). For the other sixteen — including the two families whose entire deliverable is a shrinkage-regularised quantity (`hs`/`rlm-hs` horseshoe rankings) and the disattenuated-mechanism EiV models (`mm`) — "no psense flags" in the backbone means _not measured_, not _measured clean_. This is the single most important meta-finding and it directly weakens the assurance we can give on the most prior-dependent estimands in the suite.

2. **A handful of reported estimands are genuinely prior-influenced.** These are concentrated, well-understood, and mostly honestly hedged already: the `lcf` factor correlations (all 15 flagged), the off-floor binary `itt`/`did`/`gf` treatment effects, and `did` `tau_t2` in 7/14 fits. Separately, the `mm-002` code→word slope carries a tighter scale than the linear-mechanism factory, but whether that attenuates the estimate is **untested** (see §4).

3. **Some rendered prior-panel rationales and one role label are misattributed.** The fitted `priors_table.csv` **values** match the source at the reporting-fit commit (`898ea03`) _and_ current `main` — `gamma_own` is `N(1,0.25)` in both, the distal-tier `alpha`/`tau` constructors exist (`tau_prior_distal` → `N(0,0.3)`), and the mechanism HSGPs route through `ell_prior_mech()` = `IG(5,5)` — so a refit on HEAD reproduces the tables. What is stale is a handful of **auto-derived rationale strings and one role label** (the rationale is generated from constructor docstrings via `_RV_TO_CTOR`): e.g. `mm-002` `beta_G` is tagged role `causal` with a `tau` rationale though it is an association covariate, and the `mech` `ell` rationale cites `IG(3,1)` while the fitted value is the correct `IG(5,5)`. This is a presentational bug in the rationale-derivation seam, not a value or reproducibility problem.

None of these is a blocker. All are addressable with targeted sensitivity refits, a small number of report caveats, and a source-vs-artifact reconciliation.

---

## What is working

- **Prior-predictive coverage is uniformly sound.** In every family where a `prior_predictive_check.csv` exists, the prior-predictive brackets the observed range, is appropriately wider than the data, and shows no floor/ceiling pathology. The bounded Beta-Binomial / off-floor-Bernoulli construction makes impossible sub-floor or over-ceiling counts structurally impossible. The symmetric N(0,1.5) intercept produces the expected, acceptable over-dispersion and mid-scale centring even on heavily-floored count outcomes.

- **Weakly-informative logit-scale defaults are correctly chosen and correctly scaled.** The headline `tau`~N(0,0.5) gives an 89% odds-ratio band of roughly [0.45, 2.2] — wide enough not to over-regularise a real intervention effect, tight enough to regularise at n≈55. The `gamma_cross`/association scale N(0,0.3), `gamma_A` age-as-precision N(0,0.3), and `kappa`~HalfNormal(50) are all role-appropriate. The pushforward to items/pp is credible on every test length.

- **Roles are correctly assigned** in `priors_table.csv`: causal (`tau`, `delta`, `b_M`), precision (`gamma_own`, `gamma_A`), association (`gamma_cross`, per-SD slopes), and nuisance (`alpha`, `kappa`). Age is correctly treated as a precision covariate under the empty-adjustment DAG, not licensed as causal.

- **The GP-amplitude tightening is principled.** `eta`~HalfNormal(0.3) (down from 1.0) is a sound, documented fix for the Neal's-funnel geometry that the 20-basis HSGP weights create at n≈55 (the LRP52 diagnosis). It is not crippling — the flagship L→W mechanism still yields a clearly positive +6.7-item curve, pp=0.996.

- **The regularized ("Finnish") horseshoe is textbook-correct.** `tau0`=0.1 is a reasonable fixed global-shrinkage choice, and the InverseGamma slab correctly caps effective logit slopes at ~2/SD so the Cauchy tail cannot chase absurd effects. It is, however, **approximate rather than exactly calibrated**: with `D`=9 predictors and ~51 children the p0/(D−p0)·1/√N heuristic gives `tau0`≈0.04–0.07 for p0=2–3 expected relevant predictors, so `tau0`=0.1 corresponds to roughly four active predictors — slightly less aggressive shrinkage than a strict 2–3-predictor calibration. Reporting the model-specific value (or the recommended `tau0` grid) would pin this down.

- **The mediation family gets the structural ordering right.** The b-path `b_M`~N(0,1) per SD is deliberately the _least_-regularised structural prior (so the decomposition's key path is data-led), while treatment/interaction/confounder couplings are tighter (0.5/0.3). The mean-zero a-path × mean-zero b-path product keeps the NIE prior centred at 0 and sceptical.

- **Honest hedging.** Where identification is weak (off-floor binaries, change-on-change couplings, proportion-mediated ratios), the reports already label results "inconclusive"/"suggestive" and keep unstable ratio estimands out of headlines.

---

## Prioritised concerns

### 1. Systemic: no power-scaling diagnostic for 16 of 22 families (highest priority)

**Families:** `mech` (27 fits), `med` (18), `ca` (9), `al` (9), `rlm-hg` (9), `lcsm` (5), `dose` (5), `hs` (4), `gc` (3), `mm` (3), `surv` (2), `adj`, `lcf`-analogues, `rlm-mm`, `rlm-jc`, `rlm-hs`, `rlm-adj`.

**Evidence:** psense_summary.csv is emitted only by `bx`/`did`/`gf`/`itt`/`lcf`/`lf`. The backbone's 66 flags come entirely from those six. For every other family, "no psense flags" is _absence of the test_.

**Why it matters:** the families lacking psense include the ones whose deliverable is _most_ prior-conditional:

- The **horseshoe rankings** (`hs`, `rlm-hs`): the reported P(|β|>0.1) statistic is by construction a direct function of `tau0` and `slab_scale`. The suite's single most prior-dependent object is the one family exempt from the project's prior-dominance diagnostic, and there is additionally no `tau0` sensitivity sweep.
- The **HSGP mechanism curves** (`mech`): `eta`~HalfNormal(0.3) is a deliberately tight, curve-shaping prior. Its prior-dependence was never formally checked.
- The **EiV mechanism slope** (`mm-002` `beta_code`): carries a tighter scale than the linear-mechanism factory, and its prior-dependence is untested (see §4).

**Mis-specification vs weak identification:** neither — this is a _reporting/coverage_ gap. But it converts several "clean" verdicts into "unverified." Genuinely defensible only for `rlm-mm` (fit did not converge, so psense would be uninterpretable) and, weakly, for correlation-only estimands.

### 2. `lcf` factor correlations — all 15 psense rows flagged (reported estimand)

**Parameters:** `factor_corr_pairs` (12 rows, 4 waves × 3 pairs) and `trait_share` (3 rows).

**Evidence:** prior sensitivity 0.054–0.104, likelihood sensitivity 0.202–0.336, diagnosis "potential prior-data conflict" on all 15. `trait_share` posterior pushed to 0.93–0.95 against a Beta(1.5,1.5) whose density there is ≈44% of peak. The most prior-sensitive single number is the vocabulary–grammar latent correlation (ρ=0.86, prior sens 0.104, largest disattenuation gap ≈0.26).

**Why it matters:** these ARE the family's headline deliverable, so the flags are not dismissible as nuisance. **But the direction is benign:** in every row the likelihood sensitivity is 3–6× the prior sensitivity (data dominate), and the LKJ(η=2) shrinks correlations _toward zero_, so the reported latent correlations are if anything conservative — the LKJ is actively keeping ρ(vocab,grammar) off the disattenuation ceiling, not inflating it.

**The real risk is interpretive, not the prior:** `trait_share`≈0.94 collapses the four per-wave correlation matrices into essentially one, so the wave-specific state correlations are near-prior/unidentified. **The report must not read any across-wave change in correlation structure as a finding** — it is an artifact of near-ceiling persistence.

**Mis-specification vs weak identification:** weak identification + a conservative prior; not mis-specification.

### 3. Off-floor binary treatment effects where the prior out-works the data

**Families/parameters:** `itt-011` `tau` (nonword N), `itt-009` `tau` (phonetic spelling P); `gf-005`/`gf-011` `beta_trt`; `did-011`/`did-012` `tau_t2`; and by extension `surv-009`/`surv-011` `tau`.

**Evidence (the genuinely prior-dominant cases, prior sensitivity > likelihood):**

- `itt-011`: prior 0.161 vs likelihood 0.128 — highest in the ITT family; posterior `tau_logit` 0.474 [−0.18, 1.12] barely narrower than the N(0,0.5) prior, yet the branch headline is reported **"suggestive"** positive (P(τ>0)=0.877).
- `itt-009`: prior 0.127 vs likelihood 0.093; reported "inconclusive" (P=0.72).
- `did-011`/`did-012`: `tau_t2` prior sensitivity 0.101/0.123 (highest in DiD), n_trials=1.
- `gf-005`/`gf-011`: prior 0.057/0.077 vs likelihood 0.018/0.043; reported "inconclusive."
- `surv`: `tau` posterior retains ~54–58% of its N(0,0.5) prior variance (no psense run).

**Why it matters:** `itt-011` is the one reported (albeit exploratory) _direction call_ in the suite that is meaningfully prior-dependent. A binary outcome with ~36 children, ~1 trial/child split across two arms simply cannot identify a logit contrast; the posterior leans on the prior. This is structural weak identification, not a mis-specified prior — the zero-centred prior keeps the estimate near the null with a wide interval, it does not manufacture a positive.

**Mis-specification vs weak identification:** entirely weak identification (small-n binary). The prior is behaving correctly.

### 4. `mm-002` code→word EiV slope: tighter than the linear-mechanism factory, prior-dependence untested

**Parameter:** `beta_code`, the disattenuated code→word slope, carries **N(0,0.3)** (the association scale) rather than the `N(0,1)` that `priors.py` uses for the _linear_ mechanism factory (`beta_mech`). Note `beta_mech_prior` documents that linear factory specifically, not this correlation-factor EiV slope, so there is no single "documented" prior for `beta_code` to violate — the N(0,1) comparison is a plausibility argument, not a spec breach.

**Evidence:** posterior `beta_code` = 0.34, 89% [0.09, 0.59]. An earlier draft of this review read the upper CI as "pressing against a ~0.6 (2-SD) prior ceiling," but that reasoning does **not** hold: the Normal prior is unbounded, so an interval ending near two prior SDs is not evidence of ceiling pressure or attenuation. There is **no psense and no alternative-prior fit** for this slope (only the loadings/residuals were varied, in `mm-101`), so whether N(0,0.3) attenuates the estimate is currently **untested**.

**Separately — the real reporting bug:** `beta_G` is tagged role `causal` with a `tau` rationale ("Treatment effect tau ~ Normal(0, 0.5)") in `priors_table.csv`, but `beta_G` is the **randomised-arm adjustment covariate** in an observational EiV model — an _association_, which the factory and report prose state explicitly ("never 'code drives reading'"). Its N(0,0.3) association scale is correct; the bug is only the role/rationale label, which should read `association` + the predictor-slope rationale. (Rolled into §9 / recommendation D.)

**Mis-specification vs weak identification:** neither is established. An N(0,1) refit of `beta_code` is a worthwhile **sensitivity test** — it would show whether the disattenuated code→decoding slope (a strong, well-established effect in Down syndrome) is prior-attenuated — but the current evidence does not establish that it is, nor that N(0,1) is the "correct" prior.

### 5. `did` `tau_t2` — headline causal effect flagged in 7/14 fits

**Evidence:** prior-data conflict in `did-002` (0.096/0.056), `did-010`, `did-011`, `did-012`; strong-prior/weak-likelihood in `did-001` (0.053/0.031), `did-003`, `did-013`. In the informative count outcomes the posterior sits in the prior's _right tail_: `did-002` (letter-sound) posterior median 0.595, 89% [0.20, 0.98], P>0=0.99 — the data want a larger positive effect than the zero-centred N(0,0.5) comfortably allows.

**Why it matters:** `tau_t2` is the sole reported causal quantity of the family, and it carries the _same_ tight 0.5 scale as the nuisance offsets (`beta_period`, `arm_gap_t3`). For the single causal term a marginally wider prior (N(0,0.75–1.0)) would be defensible. Magnitudes are mild and effects are still clearly recovered, so this is attenuation, not a false result.

**Mis-specification vs weak identification:** a debatable _scale choice_ on a headline term (mild), plus weak identification in the binary off-floor DiD variants.

### 6. `lf` `b_grp_time[1]` — reported t2 contrast flagged in 5/11

**Evidence:** `lf-001` (prior 0.091 vs lik 0.05), `lf-004` (0.087/0.053), `lf-005` (0.064/0.049), `lf-006` (0.052/0.042), `lf-011` (0.08/0.083). Prior pull up to ~1.8× the data's. Reassuringly, the taught-vocab outcomes where RLI had its strongest real effects are _clean_ with the likelihood dominating (`lf-009` 0.029/0.043; `lf-010` 0.043/0.06), so N(0,0.5) is not over-shrinking where data are informative. The largest raw sensitivities (`lf-005[0]`=0.147, `lf-011[0]`=0.128) sit on _unreported_ pre-randomisation/post-crossover elements.

**Why it matters:** prior-dependence appears only where the outcome is weakly identified (reading/decoding, off-floor binary). Reporting is already honest ("suggestive," CI straddling zero).

**Mis-specification vs weak identification:** weak identification.

### 7. `mech` HSGP amplitude confounds the GP-vs-linear knee tests

**Evidence:** every HSGP central estimate is ~30–50% below its linear-anchor twin for the same skill (L→W 6.68 vs 9.85; R→W 0.18 vs 2.91; E→W 0.47 vs 5.28; TR→W 3.61 vs 8.96; TE→W 1.83 vs 8.59). The GP is doubly regularised: `eta`~HalfNormal(0.3) _and_ an as-fitted `ell`=IG(5,5) (mode ~0.83, favouring near-linear curves) versus the linear slope's near-flat N(0,1).

**Why it matters:** the GP and linear specs sit on **unequal regularisation footing**, so a GP-vs-linear "knee test" partly compares prior strength, not just functional form. This may be genuine saturation (the test's purpose) but is confounded with amplitude shrinkage. Sign/existence is robust (flagship stays +6.7 items, pp=0.996); only magnitude and the functional-form _comparison_ are affected.

**Mis-specification vs weak identification:** a defensible-but-consequential prior choice; the interpretive caveat matters more than any refit.

### 8. `gf` `beta_trt` and `bx` `delta` — flagged but benign

- **`gf`:** all 5 flags on the headline `beta_trt`. Three (W, L, B) are the sceptical prior meeting a robustly positive, data-dominated effect (likelihood ≥ prior). Two (off-floor P, N) are honestly reported inconclusive. No action beyond not over-interpreting the off-floor P/N estimates.
- **`bx`:** `bx-003` (UE2) `delta` is prior-dominant (prior 0.089 > likelihood 0.077) _because of the tightened distal N(0,0.3)_ — but the estimand reports null/inconclusive (−0.28 items, 89% [−0.91, 0.38]), so it manufactures nothing. `bx-002` mild conflict, likelihood dominates 2×.

### 9. Misattributed rationale strings and one role label in `priors_table.csv` (presentational, not a value/reproducibility problem)

**Correction to an earlier draft of this review.** I initially reported a "provenance drift" between `priors.py` and the fitted tables. That was wrong, and Frank correctly pointed it out: the fitted **values** match the source at the reporting-fit commit (`898ea03`) _and_ current `main`. Specifically —

- `gamma_own_prior(sigma=0.25)` at both commits, so the tables' `N(1,0.25)` **is** the source value (there is no `N(1,0.5)` to have drifted from).
- `tau_prior_distal()` / `alpha_prior_distal()` exist and return `N(0,0.3)` / `N(0,1)`, so the "distal tier" has a real source counterpart.
- The `mech` `ell` value `IG(5,5)` is exactly what `ell_prior_mech()` returns.

A refit on HEAD therefore **does** reproduce these tables. The residual issue is narrower and purely presentational — the auto-derived rationale strings and one role label:

- `mm-002` `beta_G`: role `causal` + rationale "Treatment effect tau ~ Normal(0, 0.5)" for what is actually an association (arm-adjustment) covariate correctly scaled at `N(0,0.3)` (see §4).
- `mech` `ell`: rationale cites `IG(3,1)` (the generic `ell_prior` docstring) while the fitted value is the correct `IG(5,5)` from `ell_prior_mech()` — the rationale is derived from the wrong constructor; the value is right.
- `al` `beta_cohort` labelled "Treatment effect tau" despite explicit per-protocol/non-causal framing; `gc`/`al` `gamma_dose`/`gamma_ability` carrying verbatim `gamma_cross` text; `rlm-hs`/`rlm-adj` `beta_group_nuisance_*` rows quoting "N(0,0.3)" while the actual (and correct) prior is `N(0,1)`.
- `gc` `build_growth_model` hard-codes its sigmas inline rather than importing the shared constructors — not a drift today, but a future divergence risk.

**Why it matters:** the fitted priors are reproducible and the adequacy verdicts stand. The bug is that the rendered prior panels display rationales/roles that do not match the (correct) fitted constructors, which misleads a reader. The fix lives in the rationale-derivation seam (`_RV_TO_CTOR` mapping and the role assignment), not in any prior value.

### 10. Nuisance variance-component tensions (low severity, mostly conservative)

- **`kappa`~HalfNormal(50)** posteriors sit in/above the upper tail in `mech` (86), `lcsm` (110–190), `gc` (150–200 for E/R), `rlm-hg` (90–118 for low-noise outcomes). The prior pulls toward _more_ over-dispersion than the data want, producing the observed PPC over-coverage (e.g. `lcsm` 50%→0.84, `gc` 50%→0.76). Conservative direction, nuisance parameter.
- **`sigma_subject`~HalfNormal(0.5)** is mis-scaled _too tight_ for Down-syndrome verbal/reading heterogeneity in `rlm-hg` (DS posteriors 1.25–1.39, at/beyond the 99th prior percentile) and `rlm-jc` (word-reading 1.29 [1.06,1.56]). This is a genuine prior-data conflict on a variance component, implying a modest downward bias in estimated between-child spread. Widen to HalfNormal(1.0) for the high-variance measures.

### 11. `rlm-mm` convergence failure on a reported correlation (not a prior fault, but relevant)

The EiV cross-sectional CFA gate-failed (143 divergences, max R̂ 1.028, min ESS 64), and a **reported** number — the reading-ability factor correlation 0.881 — is itself under-converged (R̂ 1.019, ESS_tail 87). Point means are stable to ~2 dp but the 89% interval tails on these specific numbers are not trustworthy. The priors are defensible but do nothing to mitigate the near-singular 4-factor geometry; `HalfNormal(1)` on both loadings and residuals ignores the λ²+σ²≈1 budget (~32% prior mass on loadings >1, Heywood-adjacent), and the `LKJCholeskyCov` carries 4 phantom factor-SD scales that never enter the likelihood. Switching to `LKJCorr` (as the longitudinal analogue already does) would clean most of the diagnostics at no cost to any reported number.

### 12. Missing `prior_pushforward.csv` on families that DO report an estimand

`med`, `bx`, `al`, `dose`, `ca`, `lf`, `lcsm`, `surv` and the RLM families do not tabulate the prior-implied estimand in items/pp. `did` has **partial** coverage — 11 of 14 reporting fits carry `prior_pushforward.csv`; only the three dose companions (`did-006`, `did-007`, `did-107`) lack it, so the follow-up should target those specifically rather than the whole family. For `med` (a decomposition) the pushforward is the single most informative prior check and its absence is the main reason that family's otherwise-"ok" verdict carries a reporting caveat. Hand reconstruction is reassuring in every case, but the artifact should exist for parity.

---

## Per-family verdict table

| Family                   | Verdict        | Key concern                                                                                                                                            |
| ------------------------ | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `lrp-rli-itt`            | minor-concerns | Off-floor `tau` prior-dominant in itt-009/011; itt-011 "suggestive" call is prior-dependent                                                            |
| `lrp-rli-mech`           | minor-concerns | HSGP `eta` amplitude confounds GP-vs-linear knee tests (~30–50% attenuation); no psense; `ell` table self-contradiction                                |
| `lrp-rli-gf`             | minor-concerns | 5 `beta_trt` flags all benign; off-floor P/N (gf-005/011) prior-dominated but reported inconclusive                                                    |
| `lrp-rli-med`            | ok             | No `prior_pushforward.csv`; upstream dose-model convergence gate blocks IS-calibration anchor for med-059/086/087                                      |
| `lrp-rli-did`            | minor-concerns | Headline `tau_t2` flagged 7/14; carries same tight 0.5 scale as nuisance offsets; consider N(0,0.75–1.0)                                               |
| `lrp-rli-lf`             | minor-concerns | Headline `b_grp_time[1]` flagged 5/11 in weakly-identified reading/off-floor outcomes                                                                  |
| `lrp-rli-ca`             | minor-concerns | N(0,0.3) slope sits directly on the deliverable; no psense/pushforward emitted; sensitivity band only probes looser                                    |
| `lrp-rli-al`             | minor-concerns | No psense/pushforward for a family that DOES report a +2.1-item cohort estimand; rationale mislabels                                                   |
| `lrp-rli-lcsm`           | minor-concerns | No psense/pushforward/prior-pred; `b_self`~N(−0.3,0.2) is an informative off-zero prior on a reported term                                             |
| `lrp-rli-dose`           | minor-concerns | No psense/pushforward; `sigma_dose`~HalfNormal(0.5) prior-dominated over 3 periods → prior-shrunk period slopes                                        |
| `lrp-rli-hs` (horseshoe) | minor-concerns | Briefing mislabel (horseshoe, not hearing); ranking is prior-conditional yet has no psense or `tau0` sweep                                             |
| `lrp-rli-bx`             | minor-concerns | Distal `delta`~N(0,0.3) makes bx-003 prior-dominant (but reports null); no pushforward                                                                 |
| `lrp-rli-mm`             | minor-concerns | `beta_code` tighter (N(0,0.3)) than linear-factory N(0,1), attenuation untested; `beta_G` role mislabel (causal→association); LKJ geometry gate-failed |
| `lrp-rli-gc`             | minor-concerns | Growth-slope N(0,0.5) mildly tight for fastest reading trajectories; inline-hardcoded sigmas; loading positivity artifact                              |
| `lrp-rli-surv`           | minor-concerns | cloglog family; `tau` retains ~54–58% prior variance (weak ID); `alpha` reused from logit, mis-centred; no prior diagnostics                           |
| `lrp-rli-lcf`            | minor-concerns | All 15 psense rows flagged (headline correlations); benign direction; wave-specific structure unidentified                                             |
| `lrp-rli-adj`            | minor-concerns | `gamma_own` N(1,0.25), data-dominated; slope sweep robust in sign but magnitudes are prior-attenuated                                                  |
| `lrp-rlm-hg`             | minor-concerns | No prior diagnostics at all; `sigma_subject` too tight for DS verbal heterogeneity                                                                     |
| `lrp-rlm-mm`             | minor-concerns | Fit gate-failed (143 divergences); reported correlation 0.881 under-converged; Heywood-adjacent loadings                                               |
| `lrp-rlm-jc`             | minor-concerns | `sigma_subject` too tight for BAS word reading; no prior-pred/psense (verified manually)                                                               |
| `lrp-rlm-hs` (horseshoe) | minor-concerns | No psense/pushforward on the most prior-dependent model; `priors_table` rationale mislabels the group-nuisance scale (fitted value is correct)         |
| `lrp-rlm-adj`            | minor-concerns | `gamma_own` N(1,0.25), data-dominated (posterior ~0.65); slope-only sweep; rationale mislabel                                                          |

---

## Concrete recommendations

**A. Targeted prior-sensitivity refits (do these; they resolve the substantive doubts).**

1. **`mm-002` code→word slope** — refit `beta_code` at N(0,1) as a **sensitivity test** to check whether the disattenuated code→decoding slope is prior-attenuated (attenuation is currently untested, not established). Do **not** rescale `beta_G`: it is a correctly-scaled association (arm-adjustment) covariate — instead fix its role label (`causal`→`association`) per recommendation D.
2. **`itt-011` (nonword N) and, secondarily, `itt-009` (P)** — sweep `tau` over {N(0,0.3), N(0,0.5), N(0,1)} to show the "suggestive" direction is not a prior artifact. Widening will not remove the sensitivity (it is data scarcity), but it demonstrates the sign is robust. Label both branches prior-informed/exploratory.
3. **`did` `tau_t2`** — for the informative count outcomes where the posterior sits in the prior tail (e.g. did-002, letter-sound), a one-off N(0,0.75–1.0) refit to confirm the effect is not prior-attenuated. This applies only to genuine `tau_t2` DiD fits; the period-resolved dose model `did-007` has **no** `tau_t2` (its estimand is `mu_dose`~N(0,1), assigned no causal status), so it is out of scope here.
4. **`itt`/`al`/`bx` distal tier** — a single N(0,0.5) sensitivity check on the F/T and UE2 outcomes to confirm the distal N(0,0.3) tightening is not attenuating a real transfer effect (itt-025's posterior sits in the prior's upper tail).
5. **`mech` HSGP subfamily** — run psense on `eta`/`f_mech`/`beta_mech`, and either put the GP and linear specs on comparable regularisation footing or state explicitly that GP-vs-linear knee tests are not clean functional-form contrasts.
6. **`hs`/`rlm-hs` horseshoe** — add a `tau0` grid (e.g. {0.05, 0.1, 0.2}, and `slab_scale`) and show the ranking is stable. This is what substantiates the "independent cross-check" framing; without it the sub-top ranks rest on a verbal caveat only.

**B. Emit the missing prior-workflow artifacts.** Run/emit `psense_summary.csv` and `prior_pushforward.csv` for every family that reports an estimand and currently lacks them — at minimum `mech`, `al`, `dose`, `lcsm`, `ca`, `med`, `surv`, and the RLM families. For the measurement/CFA families (`lcf`, `mm`, `rlm-mm`), add an _indicator-scale_ prior-predictive check so they are not silently exempt from the coverage guarantee. `rlm-mm` is the one legitimate psense exemption (non-converged posterior).

**C. Reconsider two default priors.**

- **`sigma_subject`** — widen from HalfNormal(0.5) to HalfNormal(1.0) for the high-variance Down-syndrome verbal/reading measures in `rlm-hg`/`rlm-jc`; the current scale is in genuine conflict with the data and mildly biases the reported between-child spread downward.
- **`mm`/`rlm-mm` loadings** — replace `HalfNormal(1)` on loadings-and-residuals with a communality-scale or (0,1)-bounded loading prior that respects λ²+σ²≈1, and switch `LKJCholeskyCov`→`LKJCorr` to drop the phantom factor-SD nuisances that dominate the R̂ failures.
- Leave the core defaults (`tau`, `gamma_cross`, `alpha`, `kappa`, GP `eta`) as they are; they are correctly scaled.

**D. Fix the auto-derived rationale/role labels in `priors_table.csv` (presentational).** The fitted values already reproduce on HEAD, so there is nothing to reconcile in the prior _values_. What needs fixing is the rationale-derivation seam so the rendered prior panels match the (correct) fitted constructors: map `mm-002` `beta_G` to role `association` + the predictor-slope rationale (not `causal`/`tau`); point the `mech` `ell` rationale at `ell_prior_mech()` (`IG(5,5)`) so it stops citing `IG(3,1)`; and correct the copy-paste strings on `al` `beta_cohort`, `gc`/`al` `gamma_dose`/`gamma_ability`, and the `rlm-hs`/`rlm-adj` group-nuisance rows. Separately, move `gc`'s inline-hardcoded sigmas onto the shared constructors so they cannot silently diverge in future.

**E. Report caveats to add (cheap, high-value).**

- **`lcf`:** state that `trait_share`≈0.94 collapses the per-wave correlation matrices into one; do not interpret any across-wave change in correlation structure. Flag ρ(vocab,grammar)=0.86 as the most prior-dependent single number (LKJ holding it off the disattenuation ceiling).
- **Off-floor binaries** (`itt-009/011`, `gf-005/011`, `did-011/012`, `surv`): present treatment/hazard contrasts as prior-informed/exploratory with the weak-identification caveat.
- **`dose`:** read the period-specific slopes alongside the well-identified pooled comparator (`dose-277`); the "fading" pattern is partly HalfNormal(0.5) shrinkage.
- **`gc-070`:** the shared-factor loadings shown at prob_positive=1.0 "very strong" are a mechanical artifact of the HalfNormal positivity constraint, not evidence; the tempo↔ability correlation is null.
- **`rlm-mm`:** flag that the reported reading-ability correlation and the bpvs loading/communality are under-converged (interval tails untrustworthy), and that all memory correlations rest on an assumed reliability r=0.8.
- **`mech`:** interpret GP-vs-linear differences as confounded with amplitude regularisation, not clean nonlinearity tests.

**F. Honest bottom line on what is _not_ a problem.** The zero-centred effect priors mildly attenuating clear positive effects (the bulk of the 66 flags: itt 003/007/008/013/019/023/025/028/113/114, gf W/L/B, lcf's likelihood-dominated rows) are _desirable conservatism_, not defects — no action beyond noting them. Prior-predictive coverage needs no change. The GP-amplitude tightening was the right call for the funnel. The mediation family's structural prior ordering is exemplary. The concerns above are a short, tractable list against a fundamentally sound prior architecture.
