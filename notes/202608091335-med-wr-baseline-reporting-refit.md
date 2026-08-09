<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# MED-086/186 and MED-087/187 baseline-word-reading reporting refit

## Purpose and status

This note closes the local fit-and-audit part of issue #503 after the baseline-word-reading adjustment correction. It records the exact pinned inputs, fitted-child identities, trace-backed primary and t3 results, automatic diagnostic gates, confounding sensitivity, session-dose calibration and power-scaling caveats for MED-086, MED-186, MED-087 and MED-187. All four primary release decisions are `publishable: true`; that is a computational and artefact-completeness verdict, not a causal-identification verdict. Azure publication is deliberately recorded as pending until the authenticated upload supplies real URLs.

## Pinned run

- Run name: `20260809T110432Z-df00982f53cc`
- Local run root: `output/runs/20260809T110432Z-df00982f53cc`
- Git commit: `df00982f53cc44f3635abaef01c4c09ecf185afe`
- Fit tier: `reporting`
- Primary sampling: 6 chains, 6,000 tuning iterations and 6,000 retained draws per chain, `target_accept=0.95`, random seed 47
- Automatic clean-pass thresholds: maximum R-hat 1.01, minimum effective sample size 400, minimum BFMI 0.3 and zero divergences

The run manifest pins these seven input files:

| Input                                                           | SHA-256                                                            |
| --------------------------------------------------------------- | ------------------------------------------------------------------ |
| `LICENSE`                                                       | `9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94` |
| `reading-language-memory/README.md`                             | `2c552ff0a34bc81ad4dadc1ec61adfd75e2e3de6b3ef5911f94bddd1060dabf2` |
| `reading-language-memory/reading_language_memory_data_long.csv` | `68ea2e9c847c908b7217431af76abd45a940099ced2bfd9acf4dd69ba7e2e5f6` |
| `reading-language-memory/reading_language_memory_data_wide.csv` | `b2262d6b3b7102594b3424c4a72f4237dc84087a7b18f6fc815ccdcd0d10a55c` |
| `readme.md`                                                     | `6205d3a5e49db8a8043b15de75a35b15086f7af0c67e873c1f56c19c517f90f8` |
| `rli_data_long.csv`                                             | `dc8dda5780b705e902155372c135a993778506c547ef8ebb2b5b03668c11f043` |
| `rli_data_wide.csv`                                             | `2c47eb49a96013a0283a225dcd8460ceb62720fdca60bcaeb3811345e5b7c99c` |

## Exact fitted-child identities

The natural/interventional companions fit the same rows in the same order. The full SHA-256 identities below are computed from the ordered fitted subject IDs with the recorded domain-separated encoding; equality is exact rather than an inference from row counts.

| Models            | Rows | Unique children | Exact fitted-subject SHA-256                                       |
| ----------------- | ---: | --------------: | ------------------------------------------------------------------ |
| MED-086 / MED-186 |   50 |              50 | `9b28d8d25a6121504a1b404b72dd5a17f396af7fd47e164edfe1b7993bd686c9` |
| MED-087 / MED-187 |   53 |              53 | `f3ec82b6ef903f34f5e73fec2a692d6e5e6db6c4dae065395baf1e30c6504a57` |

The MED-086 t3 sensitivity uses 49 children and has subfit data digest `72ba744194a52900`; the MED-087 t3 sensitivity uses 53 children and has digest `0cff33a6dfa143e7`. Each is recorded once in `subfit_provenance.csv`, was sampled with the same 6 × 6,000 reporting settings, converged, and persists to its parent's `trace_mediation_t3_sensitivity.nc`. MED-186 and MED-187 do not repeat the t3 subfit by design.

## Automatic diagnostic gates and provenance

| Fit                    |      Maximum R-hat |       Minimum ESS |       Minimum BFMI | Divergences | Verdict |
| ---------------------- | -----------------: | ----------------: | -----------------: | ----------: | ------- |
| MED-086 primary        | 1.0005725102961434 |   8248.7297090968 | 0.9744543087452870 |           0 | pass    |
| MED-186 primary        | 1.0005725102961434 |   8248.7297090968 | 0.9744543087452870 |           0 | pass    |
| MED-086 t3 sensitivity | 1.0006456971810764 |  8541.76490121912 | 0.9407652252800762 |           0 | pass    |
| MED-087 primary        | 1.0006405667563940 | 9531.922800726854 | 0.9434284401431365 |           0 | pass    |
| MED-187 primary        | 1.0006405667563940 | 9531.922800726854 | 0.9434284401431365 |           0 | pass    |
| MED-087 t3 sensitivity | 1.0007618657870707 | 9420.303941080543 | 0.9337960821666947 |           0 | pass    |

The natural/interventional pairs' matching extrema follow from their identical fitted posterior in this implementation. Every primary fit has a `trace.nc`, diagnostics summary, artefact manifest, generated key findings and `release_decision.json`; the two natural fits also have the separate t3 trace and one-row structured provenance record. These checks establish stable computation and complete release inputs. They do not establish exchangeability of the mediator or resolve latent general ability, treatment-induced sessions or same-wave mediator/outcome ordering.

## Primary decompositions

Intervals are 89% equal-tailed credible intervals; estimates are posterior medians, and P>0 is the posterior probability that the contrast is positive. MED-086/186 use off-floor probability risk differences, not nonword-item changes. MED-087/187 use the 10-item phoneme-blending scale.

| Pair and estimand                                   | Direct contrast                        | Indirect contrast                      | Total contrast                         |
| --------------------------------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| MED-086 NDE/NIE and MED-186 IDE/IIE, off-floor risk | +0.005 (−0.078 to +0.090), P>0 = 0.538 | +0.079 (+0.028 to +0.148), P>0 = 0.996 | +0.087 (−0.011 to +0.188), P>0 = 0.924 |
| MED-087 NDE/NIE and MED-187 IDE/IIE, blending items | +0.441 (−0.471 to +1.366), P>0 = 0.778 | +0.292 (−0.098 to +0.761), P>0 = 0.888 | +0.748 (−0.056 to +1.560), P>0 = 0.932 |

The natural and interventional rows are numerically identical because the current implementation applies the same fitted covariate-conditional mediator law and g-formula functional. Their exact row-identity agreement and numerical equality are implementation checks; neither makes the route causal.

## Separately fitted t3 sensitivities

| Natural model and scale    | Direct contrast                        | Indirect contrast                      | Total contrast                         |
| -------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| MED-086, t3 off-floor risk | −0.085 (−0.195 to +0.024), P>0 = 0.106 | +0.103 (+0.037 to +0.186), P>0 = 0.996 | +0.022 (−0.106 to +0.148), P>0 = 0.608 |
| MED-087, t3 blending items | +0.102 (−0.846 to +1.078), P>0 = 0.569 | +0.263 (−0.144 to +0.747), P>0 = 0.854 | +0.385 (−0.447 to +1.222), P>0 = 0.772 |

The MED-086 t3 result is a binary off-floor estimand throughout. It is not compared with any superseded graded t3 item estimate, because those are different quantities. In both corrected outcome families the total interval includes zero; in the blending family the indirect interval also includes zero at both time points.

## Confounding sweep and session-dose calibration

For MED-086/186, the fitted effective mediator coefficient has mean 1.4138621345760864. The primary indirect contrast's 89% interval first includes zero at an outcome-leg shift of 0.636237960559239, or 45% of that coefficient; the posterior median at that grid point remains +0.047908917066535245 and its 89% interval is −0.006937253754568612 to +0.11491444372034555. The median remains positive through shift 1.3785155812116845 (+0.001047248672386536) and is negative at 1.4845552413048908 (−0.006204339081404192). MED-086's session-dose calibration has point shift 0.24676116262698888, below the interval tipping point, and maps it through the Bernoulli g-formula to an indirect risk difference of +0.068 (+0.017 to +0.136). The endpoint scenario spans 0.000 to 2.6559544235901074 and extends beyond both the interval threshold and the median sign change; this envelope combines separate 89% marginal endpoints and an observed-data cross-check and is not a joint credible interval. The recorded verdict is `could_account_band`: session-strength confounding could plausibly remove credible evidence for a positive indirect contrast and, at the upper end of the scenario, reverse its median. The 0.636237960559239 threshold must not be described as the point estimate becoming zero.

For MED-087/187, the fitted effective mediator coefficient has mean 0.2794318188023016, but the primary 89% indirect interval already includes zero at the unshifted fit, so there is no non-zero tipping point to report. MED-087's session-dose point shift is 0.025639469443174676; the endpoint scenario spans 0.000 to 1.204292567175874, and the mapped indirect contrast is +0.266 items (−0.127 to +0.732). The recorded verdict is `already_null`: there is no credibly non-zero indirect contrast for session-dose confounding to explain away.

## Power-scaling sensitivity

The check mark in `psense_summary.csv` means unflagged, not passed by a separate causal test. MED-086/186 flag 19 of 24 scanned parameters. Thirteen are marked for potential prior–data conflict (`a0`, `a_A`, `a_G`, `a_L`, `a_W`, `a_hs`, `b0`, `b_B`, `b_M`, `b_conf_W`, `b_deapp_c`, `b_erbto`, `kappa_M`), while six are marked for a potentially strong prior or weak likelihood (`b_A`, `b_G`, `b_GM`, `b_deapp_c_missing`, `b_hs`, `b_hs_missing`). MED-087/187 flag 9 of 22, all for potential prior–data conflict (`a0`, `a_G`, `a_L`, `a_W`, `b_GM`, `b_W`, `b_deapp_c`, `b_deapp_c_missing`, `kappa_Y`); `b_M` and the newly fitted `b_conf_W` are unflagged there. These diagnostics argue against treating clean sampler convergence or a positive median as a robustness claim.

## Interpretation

The conservative conclusion is narrower than the former stale summaries. The corrected off-floor nonword indirect contrast is positive in the primary and t3 fits, but it is model-, prior- and confounding-sensitive, and the total interval includes zero at both time points. The corrected graded phoneme-blending indirect interval already includes zero in the primary and t3 fits. Baseline-word-reading adjustment closes the newly recognised measured forks; it does not identify either decomposition. The results therefore support continued investigation of a letter-sound code route but do not establish that the intervention's nonword or blending response is causally carried through letter sounds.

## Azure publication placeholders

Authenticated Azure publication and HTTP verification are outside this note's local-edit scope. Replace each explicit placeholder only after the upload succeeds and the public object returns HTTP 200.

| Artefact        | Public Azure URL                                                         |
| --------------- | ------------------------------------------------------------------------ |
| Run/report root | `PENDING — authenticated Azure upload has not supplied a public run URL` |
| MED-086 report  | `PENDING — <public-run-root>/lrp-rli-med-086-reporting/index.html`       |
| MED-186 report  | `PENDING — <public-run-root>/lrp-rli-med-186-reporting/index.html`       |
| MED-087 report  | `PENDING — <public-run-root>/lrp-rli-med-087-reporting/index.html`       |
| MED-187 report  | `PENDING — <public-run-root>/lrp-rli-med-187-reporting/index.html`       |
