<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Attaching prior meaning at creation exposed three published priors-table defects

- **Date:** 2026-08-31
- **Status:** findings from #637 stage 2b — one is a **published reporting defect in `lrp-rli-jm-001`** and needs a decision; the other two are corrected in place
- **Issue:** #637

## What changed and why it surfaced these

Before this stage, a parameter's published **role**, **rationale** and **prior panel** were derived from its _name_: an exact-name map (`_RV_TO_CTOR`), then a name prefix, a name suffix, and in several branches the parameter's rendered distribution string. Every variable built through a named prior constructor now records a `PriorDescriptor` when it is created, so the published meaning comes from the constructor that built it rather than from what it is called.

Comparing the two answers across 36 representative models covering every family is what found the following. Where the name map and the constructor disagreed, one of them was wrong.

## Finding 1 — `lrp-rli-jm-001` publishes the wrong prior for its focal slope

The joint-mechanism **levels** design deliberately builds `beta_mech` from `predictor_slope_prior` — `Normal(0, 0.3)`, matched to `ca-010` / `ca-011` so the identified share-retained is comparable with the paired-draws ratio it replaces. The **transition** design builds the same-named variable from `beta_mech_prior`, `Normal(0, 1)`.

`_RV_TO_CTOR` mapped the _name_ `beta_mech` to the `beta_mech` constructor, so `lrp-rli-jm-001`'s stored `priors_table.csv` reads:

```
beta_mech,"Normal(0, 0.3)",association,"Linear-mechanism slope beta_mech ~ Normal(0, 1).",beta_mech
```

The `distribution` column is correct — it is read off the built variable — but the rationale names a prior three times wider than the one fitted, and the `panel` column points the report at `prior_beta_mech.png`, which plots that wider density. A reader checking whether the flat fitted slope is evidence or prior shrinkage is shown the wrong prior. `lrp-rli-jm-002` (transition) is unaffected and correct.

**Nothing about the fitted model is wrong** — the posterior, the gate and every estimate stand. This is a reporting defect in one row of one table and the panel it references.

**Decision needed:** whether to regenerate `jm-001`'s `priors_table.csv` (and its prior panel) from the stored trace without refitting, or to leave it until the next reporting refit. The table is regenerable without sampling.

## Finding 2 — the mechanism lengthscale, fixed twice

`f_mech__ell` carried the same defect #586 finding 3 described: the `__ell` suffix rule routed every mechanism lengthscale to the generic `ell` constructor, so the report panelled `InverseGamma(3, 1)` for a curve fitted under `InverseGamma(5, 5)` or `InverseGamma(8, 8)`. #586 repaired it by adding `ell_mech` / `ell_mech_tight` constructor keys and having `prior_artifacts` select one **from the resolved run plan**.

The descriptor gets it right without that indirection, because the factory records which constructor it actually called. The run-plan override still applies and still wins, so no stored fit changes; the override is now redundant rather than load-bearing, and a direct factory caller (a test, a probe) gets the right answer too.

## Finding 3 — age couplings were a precision term only by name

`gamma_A`, `a_A`, `aL_A`, `aE_A`, `aB_A` and `b_A` are age couplings. Age is a **precision** covariate in this workflow, and the name-based classifier said so through a `_A$` rule, checked deliberately ahead of the `Normal(0, 0.3)` signature test with the comment "age shares the cross-coupling scale but is a precision term, not an association".

Eleven call sites built them from `gamma_cross_prior()` — the association constructor. `gamma_age_prior()` and `gamma_cross_prior()` return the **identical** distribution, `Normal(0, 0.3)`, so those sites now call the age constructor instead. The registered distribution is byte-identical, so no posterior, prior-predictive or sampled quantity moves; the code now declares what the name rule was compensating for.

## What is still inferred

The migration is half done by variable count. Of 147 distinct free-variable names across the representative models, **83 are built through a named constructor** and now carry descriptors; **64 are inline `pm.*` calls** and still reach their published role through the name-and-scale classifier.

That half cannot be finished by the same mechanism alone: two of those variables — the HSGP basis coefficients `f_A__g_unit_hsgp_coeffs` and `f_mech__g_unit_hsgp_coeffs` — are created **inside `dse_research_utils`**, which this repository cannot annotate at the creation site. So "attach the meaning where the variable is created" cannot be an absolute rule; a small explicit declaration table is needed for externally-created variables however far the migration goes.

The remaining ~50 in-repo inline sites are all in `factories.py`, which #637 stage 3 splits by family. Declaring them is cheaper after that split than before it, and doing it first would collide with it.

## Finding 1 resolved — 2026-08-31

`lrp-rli-jm-001-reporting`'s prior table was regenerated in place with `scripts/regenerate_priors_table.py`, and the report re-rendered. **No resampling**: the prior table is a property of the model's structure, so the script rebuilds the model from the fit's own recorded plan — including the artefact-hosting wave `config.json` names — checks the rebuilt free-variable set against the stored table, and only then writes.

One line changed:

```
- beta_mech,"Normal(0, 0.3)",association,"Linear-mechanism slope beta_mech ~ Normal(0, 1).",beta_mech
+ beta_mech,"Normal(0, 0.3)",association,"Standardised predictor slope ~ Normal(0, 0.3) by default.",predictor_slope
```

`distribution` and `role` were already right. `prior_beta_mech.png` / `.svg` were deleted — they plot `Normal(0, 1)`, a density this model does not use — and the two matching `untracked` rows were pruned from `artifact_manifest.json`. Every one of the 24 recorded manifest rows is byte-identical, nothing else in the directory changed, and the re-rendered `index.html` now shows the corrected rationale and the `predictor_slope` panel.

Two things the regeneration deliberately does **not** do, each because the first attempt did and it was wrong:

- **It does not redraw the panels that were already correct.** `emit_priors` redraws every panel, and a panel redrawn now is laid out by today's matplotlib: same density, different canvas dimensions, four unrelated figures diverging from the rest of the corpus for no reason.
- **It does not rescan the manifest.** A published fit directory also holds a rendered `index.html` and its Quarto asset tree, written after the fit. A full rescan folds those in, turning the fit's own record of what it wrote into a listing of what happens to be in the directory.

A related defect surfaced while doing this and is fixed in the same change: `priors.used_prior_keys` still resolved each variable's panel through the name map while `priors_table` resolved it from the recorded descriptor. The two now share one `described_prior_row`, so a row and the figure beside it cannot name different densities.
