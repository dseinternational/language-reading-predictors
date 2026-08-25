<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# The aligned and concurrent blending link pairs, and the helper that was missed twice

- **Date:** 2026-08-25
- **Status:** implementation record
- **Issue:** #619 (under the #608 policy), first instalment

## What was built

Two more families now carry the mandatory phoneme-blending response-link pairing:

| Family       | Pair                                | Legacy id  | Below-chance mass in the ordinary fit |
| ------------ | ----------------------------------- | ---------- | ------------------------------------: |
| `aligned`    | `lrp-rli-al-006` + `lrp-rli-al-306` | `lrpal06f` |                                 4.9 % |
| `concurrent` | `lrp-rli-ca-007` + `lrp-rli-ca-307` | `lrpca07f` |                                 9.7 % |

Each gets `score_mean_link` in its typed settings, resolved run plan and factory, with the same B-only contract validated before any I/O, plus a payload carrying what the factory _built_ so the summaries cannot drift from the likelihood. `score_mean_link="logit"` delegates to the previous call unchanged, so every other fit in both families is byte-identical.

The `f` link-sensitivity suffix takes the **3xx block in every family**, so a `-3NN` id reads as "the link companion of `-0NN`" wherever it appears. That is a deliberate departure from the per-family "next free block" habit (`gf` has `b`→1xx and `m`→2xx; `lf` has `b`→1xx and `a`→2xx): for a brand-new suffix it costs nothing to fix one block from the start, and it makes the role legible across families.

## Neither pair binds a causal quantity, and that is the point

Both families are wholly associational. The aligned cohort contrast is confounded by age-at-onset and cohort/timing; every concurrent coefficient conditions on contemporaneous post-treatment skills by design. Under the pre-#608 reading, that would have been the argument for exempting them.

#608 rejected it, and correctly. The link determines the mapping from the latent scale onto the natural one, so **any quantity published in items inherits it regardless of what identifies it**. `cohort_marginal.csv` and `concurrent_marginals.csv` are both published in items. A latent-logit coefficient's _sign_ is more robust to the link than a natural-scale marginal is — which is an argument for reporting latent-scale quantities where they answer the question, not for leaving a misspecified link under a natural-scale headline.

## Two family-specific wrinkles

**The concurrent card is a table.** Every other paired family publishes one headline row — a ROPE summary, a DiD t2 level, a cohort marginal. The concurrent family publishes one marginal per wave × predictor × scale and names no single card. Rather than invent a "headline row" the family does not define and put a number in the audit record that no report ever shows, `_StoredPairSpec.card_columns` is now optional: when empty, the pair check verifies the table exists, is non-empty, and has the **same number of rows** on both sides, alongside the usual data / fitted-rows / sampling-config identity. Nothing about the bindingness changes — what binds a pair was always the identity evidence, not the card values.

**The concurrent link governs blending as the _outcome_ only.** `ca-001` through `ca-006` carry B among their _predictors_, where it enters as a standardised same-wave logit covariate rather than as a modelled score mean. There is no B score mean in those fits to floor, so the resolver rejects the floor link for any non-B outcome and those six models are untouched. An association _with_ blending is not a blending score.

Scope in `aligned` is the model of record: the `al-101` cumulative-session dose variant conditions on a collider and is a sensitivity reported beside the headline, so it is exempt on the same boundary the level family's window comparator and the gain family's variants draw. Its resolved design names the paired headline rather than going silent. The concurrent family has no variant role, so every B fit in it is a model of record.

## The helper that was missed twice

`concurrent_marginals` differenced raw inverse-logits with no link parameter. So did `marginal_prior_pushforward`, which the concurrent, adjusted, horseshoe and dose companions all use for their prior checks. Both are fixed.

`concurrent_marginals` is worth recording because of _how_ it was missed. Sizing this work, I mapped signature line numbers to function names with a script that took the nearest preceding `def` — and the `score_mean_link` at what was then line 4729 belongs to `level_t2_marginal_effect`, which sits _after_ `concurrent_marginals` in the file. The lookup reported `concurrent_marginals` as already link-aware, I wrote that into the sizing table, and the pipeline only failed at the `TypeError` when the dev fit ran. Had the parameter existed but been ignored, nothing would have failed at all.

`tests/statistical_models/test_blending_sensitivity.py` now pins the inventory: every natural-scale reporting helper — the twelve that turn `eta` into probability/items output — must accept `score_mean_link`. A new one has to opt in explicitly rather than defaulting to a hidden ordinary-link assumption. The lesson generalises past this issue: a script that infers structure from line numbers is evidence about the script, not about the code.

## Verification

- Both companions build on the same rows as their primaries with identical free random variables; only the observed node's link differs.
- Both pipelines run end-to-end at `dev`, and both are fitted at `reporting`.
- The release gate withheld `lrp-rli-al-006` and `lrp-rli-ca-007` from the moment their branches landed until their twins were fitted, and left the non-B siblings, the `al-101` dose variant and the four other families untouched.
- Full suite green; 23 new tests.

## Still outstanding on #619

`dose_response` and `mediation` still have no pairing: LRPDOSE84, LRPMED87 and LRPMED187 publish unpaired, and neither family declares `score_mean_link`. `mediation` is the substantial one — the link has to enter the g-formula's counterfactual simulation in all three `decompose*` functions, not merely a summary — and it is the family where the model-of-record scope question returns, because MED-187 is a numerically identical `interventional` relabelling of MED-087.

The two cross-cutting items are also open: keying the gate on `outcome_symbol` rather than `kind`, and #608's decision 2, that every pair should be bound by the content-addressed archive rather than the two-directory stored-artefact check.
