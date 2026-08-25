<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# The gain family's phoneme-blending link pair, and why its scope stops at the model of record

- **Date:** 2026-08-25
- **Status:** implementation record + one scoping decision
- **Issue:** #596 (under the #608 policy; also closes acceptance item 7 of #575)

## What was built

`lrp-rli-gf-006` fitted the ordinary Beta-Binomial inverse-logit score mean for a ten-item, **three-alternative forced-choice** test and had no guessing-floor companion. It now has one: `lrp-rli-gf-306`, the same model under `mu = 1/3 + (2/3) * inverse_logit(eta)`. Neither may be released without the other.

The family plumbing that makes the pair possible — `score_mean_link` in `GainFactorsModelSettings`, the resolved run plan, `build_gain_factors_model` and every natural-scale summary the pipeline derives — is new in the same change. The link functions themselves (`apply_score_mean_link`, `invert_score_mean_link`, `beta_binomial_from_score_mean_link`) were already shared and family-generic.

## The evidence this rests on

The #608 note measured how hard the floor binds in each unpaired `B` fit. Those figures reproduce exactly against the stored `reporting` traces, including the independently-recorded LRPLF06 cross-check (24/215) that validates the method:

| Fit      | Rows with posterior-mean expected proportion < ⅓ | Posterior mass below ⅓ | Worst single row |
| -------- | ------------------------------------------------ | ---------------------- | ---------------- |
| LRPGF06  | 15/161                                           | 10.7 %                 | 99.8 %           |
| LRPGF06m | 13/161                                           | 10.5 %                 | 99.9 %           |
| LRPGF06b | 4/135                                            | 8.4 %                  | 91.5 %           |

For scale, LRPITT08 — the fit the policy was written for, and the only pair fitted under both links — carries 8.9 %. The gain primary binds _harder_ than the ITT fit whose companion halved its estimate. There is nothing marginal about this one.

Reproduction: `eta` from `trace.nc`, `mu = expit(eta)`, count rows whose posterior-mean `mu` is below `1/3` and draws whose `eta` is below `-log 2`.

## The scoping decision: the model of record only

**Decision.** The pairing binds `lrp-rli-gf-006` (+ `lrp-rli-gf-306`). The treated-only companion `lrp-rli-gf-106` and the moderation variant `lrp-rli-gf-206` are **exempt**, under the recorded-and-dated exemption clause of the #608 decision. This note is that record.

**Why.** #608 settled that the causal/observational split does not earn an exemption, and it does not — but the axis here is different. The pairing governs _the fit whose `B` card is published as this family's blending headline_, and that is the interaction-free graded primary. Three things point the same way:

1. It is the boundary the repository already draws. `release.gate_applies` skips gain treated-only companions (no `beta_trt` exists to gate) and gain moderation variants (their interaction-aware marginal is by decision never released as causal, #391 finding 3). Adding a second, differently-drawn boundary for the same two models would make the release rules harder to reason about, not safer.
2. It is the boundary the level family already drew, in shipped code, for the same reason. `level_factors.py`'s `model_of_record_window` exempts the window comparator LRPLF06a from the pairing, and its comment cites the gain family's moderation-variant exclusion as its precedent. Requiring floor twins of gain variants while exempting a level comparator on identical reasoning would be incoherent.
3. Fail-closed has to do work, not damage. Requiring the pairing of a variant with no floor twin withholds a published fit in exchange for nothing a reader gains — the link question for blending is answered, at full strength, by the pair beside it.

**What the exemption does not do.** It does not hide the link question from those reports. Both variants' resolved plans now carry a sentence naming the paired headline — "This outcome's published blending estimate is the link-paired headline (`lrp-rli-gf-006` + `lrp-rli-gf-306`) … this variant carries the ordinary inverse-logit score mean alone, so it answers its own variant question and not the response-link one" — so a reader of LRPGF06b or LRPGF06m is told where the link-checked number lives. That mirrors how the level window comparator handles it.

**Tension with the #608 note, stated plainly.** The #608 note's "what this implies" section lists LRPGF06b, LRPGF06m and LRPLF06a among eight models needing companions. That list reads the scope rule literally; the shipped level code and `gate_applies` already read it as model-of-record. This note settles the gain family on model-of-record and flags the level comparator's identical status as consistent, not as a defect. If a later decision reverses this, it reverses all three together.

## Two structural questions the checklist asked

**Is there an empirical-Bayes intercept location to map back through the link?** **No.** The level companion had to (`invert_score_mean_link`; its anchor moved 1.1 logits). `_alpha_sigma_for` tiers the gain intercept prior's **SD** and leaves `mu = 0.0`, because a gain ANCOVA carries the outcome level in `gamma_own * logit(y_pre)` rather than in the intercept — its own docstring says so ("not re-anchored"). There is nothing to remap, and adding an anchor here would be a second change to a fit whose purpose is to isolate the first.

**Does `gamma_own` need adjusting?** It is **deliberately left alone**, and this is the one assumption worth flagging. Its `Normal(1, 0.25)` prior is centred on "post-logit tracks pre-logit 1:1". Under the floor link `eta` is the logit of the rescaled mean `(mu - 1/3) / (2/3)`, not of `mu`, so that 1:1 reading is no longer exact. `lrp-rli-itt-108` — the archive-grade pair, with the same own-baseline structure — keeps it unchanged, and a link sensitivity that also moved a baseline prior would stop isolating the link. Recorded in the module docstring and the report so the assumption is visible rather than implied.

## The natural-scale summaries, which were the real hazard

The #608 note called this out as following from the decision rather than being discretionary, and it is the part that would have failed quietly. `treatment_marginal_effect` and `association_marginals` differenced raw inverse-logits with no link parameter at all. Left as they were, `lrp-rli-gf-306` would have sampled a floor-link posterior and then published ordinary-link items from it — a wrong number wearing the right label, in `treatment_marginal.csv`, `association_marginals.csv`, `rope_summary.csv` and the predicted-score panels alike.

Both helpers now take `score_mean_link` and map both counterfactual arms through it before differencing. The pipeline reads the link from the **fitted payload** — what the factory built — rather than from the declared setting, so the summaries cannot drift from the likelihood. `score_mean_link="logit"` delegates to the previous call unchanged, so every other gain model builds and summarises byte-identically.

## What the pair says

`lrp-rli-gf-306` is fitted at reporting tier. It sampled cleanly — 0 divergences, maximum R-hat 1.0012, minimum ESS 7,613 — on the **same 161 rows and 54 children** as LRPGF06 (identical `fitted_data_identity` digest `ad7c861af4c22af5` and identical `data_sha256`), so the comparison isolates the link.

|                               | LRPGF06 (logit) | LRPGF06f (guessing floor) |
| ----------------------------- | --------------: | ------------------------: |
| Period-1 effect, items        |           +0.84 |                 **+0.50** |
| 89 % credible interval, items |  +0.09 to +1.58 |        **−0.06 to +1.04** |
| P(effect > 0)                 |           0.963 |                     0.923 |
| P(benefit ≥ δ = 1 item)       |           0.359 |                 **0.071** |
| P(practically negligible)     |           0.641 |                 **0.929** |
| `beta_trt`, logit scale       |           +0.39 |                     +0.47 |

**The link changes the finding, not the scale.** The items estimate falls by about 40 %, the 89 % interval crosses zero, and the practical verdict inverts: under the ordinary link there is a 36 % chance of a benefit of at least one item, under the floor link 7 %. The latent coefficient moves the _other_ way (+0.39 to +0.47) — as it must, since the floor link compresses a given latent shift into a smaller response-scale change — which is exactly why a natural-scale headline cannot be read off the wrong link.

This reproduces the pattern the ITT pair showed (+0.99 → +0.49, interval crossing zero, evidence label dropping a rung), on an independent family, likelihood and row set.

**And the floor link fits marginally better.** PSIS-LOO gives `elpd_loo` −324.24 (SE 9.43) for the ordinary link and −321.75 (SE 9.92) for the floor link, a difference of **+2.49 (SE 3.55)** in the floor link's favour, with all 161 Pareto-k below 0.56 in both fits. Comfortably inside noise, but not worse — the same verdict as the ITT pair's +1.09 (SE 2.68). The constraint costs nothing in fit and is mechanically motivated by the test's own design, so on this family too it is the ordinary-link number that needs justifying, not the floored one.

Both halves now pass the release gate and publish together.

## Verification

- The pair is a genuine pair: both halves build on the same 161 rows and 54 children with **identical free random variables**; only the observed node's link differs.
- Before `lrp-rli-gf-306` was fitted, the release gate withheld the stored `lrp-rli-gf-006-reporting` fit — "the paired `lrp-rli-gf-306` fit is not present beside this one … fit the pair before releasing either side" — while `lrp-rli-gf-106` and `lrp-rli-gf-206` are unaffected, and the four other families' unpaired `B` fits (LRPAL06, LRPCA07, LRPDOSE84, LRPMED87) are unchanged, as scoped.
- gf-006's stored plan predates `link_sensitivity_required_for_release`; the gate derives the requirement from the registered ids as well, so a stale stored plan cannot bypass it. Tested.
- Full suite green.

## Not done here

The gate's family dispatch is still keyed on `kind`. `aligned`, `concurrent`, `dose_response` and `mediation` still have `B` models — LRPAL06, LRPCA07, LRPDOSE84, LRPMED87 — that publish unpaired, and their families still need `score_mean_link` in settings, run plan and factory. #608's second decision, that every pair should be bound by the content-addressed archive apparatus rather than the two-directory check, is also outstanding: this pair uses the same stored-artefact tier as the level and DiD pairs. Those are the remaining #608 work, not #596's.
