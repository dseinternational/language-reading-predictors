<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# The mediation blending link pair completes #608, and it moves the decomposition

- **Date:** 2026-08-25
- **Status:** implementation record
- **Issue:** #619 (under the #608 policy), final family

## What was built

`mediation` is now paired: `lrp-rli-med-087` + **`lrp-rli-med-387`** (legacy `lrp87f`). With it, **all eight `B` families carry the pairing** the #608 decision requires — `itt`, `level_factors`, `did`, `gain_factors`, `aligned`, `concurrent`, `dose_response`, `mediation`.

The release gate's default is now **keyed on the outcome symbol** rather than on `kind`, which is what #608 decision 1 always said and the code never did. Before #619 the dispatch returned early for every unlisted kind, which is how four families published unpaired `B` results for months without anything failing. A `B` fit in a family with no registered pair gate now fails closed, and a test asserts no registered `B` model relies on that default.

## Why this family is different

In every other paired family the link corrects a **summary** — a treatment marginal, a ROPE row, a table of association marginals — computed once from the posterior. Here it cannot.

Every NDE, NIE and total is a **difference of simulated outcome means**. The g-formula accumulates `E[Y(g, M(g'))]` over units and mediator replicates and then differences the cells; there is no downstream number to correct afterwards. So `score_mean_link` is threaded into `decompose`'s `outcome_p` — the single place the latent scale becomes a score — and every counterfactual cell is accumulated on the response scale the outcome likelihood used.

It governs the **outcome** only. The mediator here is letter sounds: a different measure with its own leg and its own denominator. No registered mediation model has phoneme blending as its mediator, and the resolver rejects the floor link for any non-B outcome. The parameter was added to all three `decompose*` entry points even though only the single-mediator path has a `B` model today, because a future two-mediator or period-stacked blending fit would otherwise inherit the ordinary link silently — which is the exact failure mode this whole issue exists to close.

## What the pair says — the link changes the structure, not just the size

`lrp-rli-med-387` sampled cleanly: 0 divergences, max R-hat 1.0007, min ESS 9,317, on the same rows as the primary.

| Quantity | LRP87 (logit) | LRP87f (guessing floor) | P(> 0)        |
| -------- | ------------: | ----------------------: | ------------- |
| Total    |  +0.674 items |              **+0.486** | 0.908 → 0.890 |
| NDE      |        +0.411 |              **+0.207** | 0.763 → 0.709 |
| NIE      |        +0.247 |              **+0.267** | 0.842 → 0.882 |

**The direct effect halves while the indirect effect does not move.** That is a different result in kind from the rest of the #619 series, where the link shrank magnitudes roughly uniformly or shuffled inconclusive terms. Here the _balance between the two routes_ changes: under the ordinary link the direct route carries the larger share of the total, and under the floor link the mediated route does.

The posterior median proportion mediated moves from **0.30 to 0.47**, and that number should be read as a description of the shift and nothing more — its 89 % interval is −1.07 to 2.32 under the ordinary link and −1.22 to 2.49 under the floor link. A ratio whose interval spans well outside [0, 1] is not a usable summary, which is why the pipeline already declines to publish it on the items scale (`words_median` is deliberately `NaN` for that row). The finding worth stating is the NDE halving, not the proportion.

**Why this happens, mechanically.** The floor link compresses the response scale from (0, 1) onto [1/3, 1], so a given latent shift maps to a smaller change in expected items — but not by a constant factor, because the compression bites hardest where the ordinary link was predicting below chance. The direct contrast moves children across exactly that region; the mediated contrast operates through the mediator's fitted law and largely does not. So the two routes are not rescaled equally, and the decomposition shifts.

**What it does not change.** None of the identification. The binding unverifiable assumption is still no unmeasured L → B confounding, which latent general ability violates; intervention sessions remain a treatment-affected recanting witness that no adjustment set rescues. This is a model-based g-formula decomposition under stated cross-world assumptions, wide at n ~ 53, not an identified natural effect. The link fixes the response scale; every caveat that stood before stands now.

Worth recording alongside: LRP87's ordinary-link posterior carries the **largest below-chance share of any registered `B` fit** — 11.6 % of its row-by-draw mass, above LRPITT08's 8.9 %, the fit the policy was written for. Of all the fits #608 identified, this was the one most exposed to the defect, and it is the one where the link moved the answer most.

## Scope

The model of record, as everywhere else in #619. `lrp-rli-med-187` declares `companion_of` and is, by this family's own contract, an `interventional` relabelling whose numbers reproduce LRP87's exactly, so it is exempt on the boundary the level window comparator, the gain variants and the aligned dose sensitivity already draw. Its prose names the paired headline.

## The retargeting treadmill, ended

`test_key_findings.py`'s "a `B` outcome in an unpaired family still finalises" test was retargeted three times across #619 — `aligned` → `dose_response` → `mediation` — because each retarget was its subject being paired. With every family paired there is no unpaired family left to name, so it has been replaced by two tests that pin what actually matters now: that a non-ITT `B` fit does not need the ITT _archive_ bundle (using `med-187`, a legitimately unpaired exempt fit), and that **every registered `B` model is either in a link pair or a recorded variant of one**.

## Verification

- Both halves build on the same rows with identical free random variables; only the outcome node's link differs.
- The pipeline runs end-to-end at `dev` and `reporting`; the pair evaluates release-ready.
- The gate withheld `lrp-rli-med-087` from the moment its branch landed until the twin was fitted, and left `med-187`, `med-059` and the rest of the family untouched.
- The symbol-keyed default was checked in both directions: a synthetic `pooled_levels` fit with a `B` outcome fails closed; the same family with a `W` outcome does not.
- Full suite green; 12 new tests.

## What remains of #608

One item: decision 2, that every pair be bound by the **content-addressed archive** apparatus rather than the two-directory stored-artefact check. Today only the ITT pair is archive-bound; the other seven use the lighter tier, and a rendered report does not show a reader which tier backs a given card — which is the ambiguity the policy exists to remove. It needs `blending_sensitivity`'s archive parameterised by pair and by per-family focal columns; it is currently hardcoded to one global pair with ITT-shaped free variables (`alpha`, `tau`, `gamma_own`, `gamma_A`, `kappa`) and an ITT-shaped summary-column map. The per-family `_StoredPairSpec` declarations added across #619 are the natural input to that parameterisation.
