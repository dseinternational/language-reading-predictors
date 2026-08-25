<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# The aligned, concurrent and dose blending link pairs, and the helper that was missed twice

- **Date:** 2026-08-25
- **Status:** implementation record
- **Issue:** #619 (under the #608 policy), first instalment — three of the four unpaired families

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

## What the two pairs say

Both companions are fitted at `reporting` tier on the same rows as their primaries, and both sampled cleanly: al-306 with 0 divergences, max R-hat 1.0003, min ESS 20,739; ca-307 with 0 divergences, max R-hat 1.0005, min ESS 23,681.

**Aligned** moves the way the earlier pairs did. The per-protocol cohort marginal falls from **+0.30 items** (89 % −0.58 to +1.20, P(>0) 0.706) to **+0.18** (89 % −0.52 to +0.88, P(>0) 0.655). Neither number was ever evidence of anything — both intervals sit squarely across zero, as a confounded per-protocol contrast should — but the ordinary-link version was about 70 % larger.

**Concurrent is the one worth reading carefully, because the link is not a rescaling.** Its t4 adjusted `+1 SD` marginals:

| Predictor               | Ordinary logit | Guessing floor | Ratio | P(>0)         |
| ----------------------- | -------------: | -------------: | ----: | ------------- |
| Word reading            |          +0.82 |          +0.53 |  0.64 | 0.996 → 0.986 |
| Letter sounds           |          +0.43 |          +0.38 |  0.90 | 0.919 → 0.953 |
| Taught receptive vocab  |          +0.04 |          +0.10 |  2.50 | 0.553 → 0.665 |
| Taught expressive vocab |      **−0.04** |      **+0.15** | −3.71 | 0.452 → 0.725 |
| Receptive vocab         |          +0.33 |          +0.16 |  0.49 | 0.865 → 0.744 |
| Expressive vocab        |          +0.24 |          +0.24 |  0.97 | 0.767 → 0.823 |

The headline association shrinks by a third, receptive vocabulary halves, expressive vocabulary barely moves — and taught expressive vocabulary **changes sign**. Nothing here is decisive (every one of those small terms has a probability of direction well inside the inconclusive band, so the sign flip is noise being relabelled rather than a finding reversing), but it makes the mechanism concrete: the floor link compresses the response scale non-uniformly across rows, so a _ranking_ of items-scale associations is link-dependent in a way a single treatment contrast does not reveal. A reader comparing predictors by their items-scale marginal is comparing something the link partly determines.

That is the clearest argument yet against the "it's only an association, so the link matters less" intuition that #608 rejected: in this family the link's effect on the reported quantities is _larger and less predictable_ than in the causal families, not smaller.

## Verification

- Both companions build on the same rows as their primaries with identical free random variables; only the observed node's link differs.
- Both pipelines run end-to-end at `dev`; both companions are fitted at `reporting` and pass their gates, and both pairs now evaluate release-ready.
- The release gate withheld `lrp-rli-al-006` and `lrp-rli-ca-007` from the moment their branches landed until their twins were fitted, and left the non-B siblings, the `al-101` dose variant and the four other families untouched.
- Full suite green; 23 new tests.

## The dose family, added in the same branch

`dose_response` is now paired too: `lrp-rli-dose-084` + **`lrp-rli-dose-384`** (legacy `lrp84f`). The companion sampled cleanly — 0 divergences, max R-hat 1.0011, min ESS 8,832 — on the same rows as the primary, with `target_accept` matched at 0.97 so the comparison isolates the link rather than confounding it with a sampling-quality difference.

This is the family the #608 decision named as the case that defeats the observational exemption, so the pairing matters more for what it settles than for what it moves. `METHODS.md` defines every dose fit's focal estimand as the **natural-scale treated-row dose marginal**, published in items by `dose_marginal_summary.csv`, so it inherits the link exactly as a randomised contrast does. That no dose slope is causal changes what the number means, not what scale it sits on.

|                                  |    LRP84 (logit) | LRP84f (guessing floor) |
| -------------------------------- | ---------------: | ----------------------: |
| Treated-row dose marginal, items |           +0.017 |                  −0.027 |
| 89 % credible interval           | −0.740 to +0.724 |        −0.655 to +0.639 |
| P(> 0)                           |            0.515 |                   0.472 |

**Read honestly, this fit says nothing under either link, and the pairing does not change that.** The point estimate crosses from a hair above zero to a hair below it, but P(>0) moves 0.515 → 0.472 — both squarely inconclusive — and the interval merely narrows slightly. The sign flip is noise being relabelled, the same phenomenon as the concurrent taught-expressive term above. What the pair buys here is audit consistency: a reader can now see that the one blending dose result is inconclusive under _both_ links, rather than having to wonder whether the ordinary link was holding it up.

One implementation detail is worth recording because it prevents a subtle failure. `dose_marginal_draws` is the single transform behind **both** the posterior dose marginal and its prior pushforward — an invariant #587 finding 5 established so the "check" could not compare two different contrasts. Passing the link into that one helper extends the invariant to the response scale: the prior check cannot end up on a different scale from the posterior it checks.

A stale qualifier is also cleared. `docs/models/README.md` described `dose-084` as "qualified — the required guessing-floor link companion is not yet built for this family". It is built.

## Still outstanding on #619

`mediation` is the **last** family with no pairing: LRPMED87 and LRPMED187 publish unpaired and the family does not declare `score_mean_link`. It is the substantial one — the link has to enter the g-formula's counterfactual simulation in all three `decompose*` functions, not merely a summary — and it is where the model-of-record scope question returns, because MED-187 is a numerically identical `interventional` relabelling of MED-087.

Note for whoever takes it: `test_key_findings.py::test_unpaired_family_blending_outcome_does_not_require_the_itt_bundle` has now been retargeted twice, from `aligned` to `dose_response` to `mediation`, because each retarget was a family becoming paired. When `mediation` is paired there is no valid subject left, and the test should be replaced by an assertion that _every_ registered `B` model is paired.

The two cross-cutting items are also open: keying the gate on `outcome_symbol` rather than `kind`, and #608's decision 2, that every pair should be bound by the content-addressed archive rather than the two-directory stored-artefact check.
