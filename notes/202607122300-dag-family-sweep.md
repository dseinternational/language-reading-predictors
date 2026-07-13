<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# DAG-dependent family sweep — the #247 second stage (adjusted / dose / lcsm / growth)

Date: 2026-07-12

Related: #247 (parent), the PR-1 re-derivation (`notes/202607122200-gf-lf-revised-dag-adjustments.md`), #251 (DAG-revision tracker), #250/#264 (the time-lagged DAG this defers to).

The #247 second stage re-checks the remaining DAG-dependent families against the revised 2026-07-10 DAG (`dag/dag-language-reading.dagitty`). The headline finding: **three of the four families need no adjustment-set change**, and the fourth (`adjusted`) gains the new upstream traits as tested covariates. The rest is prose de-staling (the old graph's `notes/202606231600-…` citation and "locked DAG" wording).

## The re-check, per family

### `dose_response` (DOSE-077 / 177 / 277) — no change

The exposure is intervention sessions `IS`; the outcome is word reading `W`. `IS`'s parents in the revised DAG are `{A, GA, IG}` — and crucially **HS/SP/RW are not ancestors of `IS`** (HS is a root; SP and RW have parents `{A, GA, HS}` only), so they cannot confound `IS→W`. The observable back-doors `IS←IG→W` and `IS←A→W` are blocked by `{G, A}`; the `IS←GA→W` path is latent and unadjustable in both the locked and the revised graph — which is exactly why the dose slope is reported as an **adjusted association, not identified**, and why the ability-adjusted sensitivity fit DOSE-177 (conditioning on the baseline-skill cluster L/E/B as `g`-proxies) exists. The revision does not touch `IS`'s parents, so `{G, A, W_pre}` stands. (Prose fix: the docstrings said the DAG "omits `g→dose`"; corrected to say the model's identification treats arm `G` as the sole confounder while the revised base graph carries the latent `GA→IS`.)

### `growth` (GC-069 / 070) — no change

The estimand is a descriptive `NVMA(blocks)→trajectory-shape` association. Block design is an **off-DAG ability proxy**, so the revised graph introduces no new back-door to block; the model is already framed as a GA-confounded adjusted association. Only the `locked DAG` citation was stale.

### `lcsm` (LCSM-067) — no change

The headline couplings `g_L`/`g_E` are **within-child** (prior-wave latent L/E level → subsequent reading change). HS/SP/RW are stable **between-child** traits; within a child they do not vary, so they cannot confound a within-child coupling, and the latent true-score structure already separates measurement from dynamics. Only the citation was stale.

### `adjusted` (ADJ-065) — HS/SP/RW added as tested covariates

ADJ-065 is a between-child mutually-adjusted regression of word-reading gain on the baseline-skill cluster (letter sounds, a language composite, blending), already _testing_ whether non-verbal MA / behaviour carry independent signal. The revised DAG places three new upstream traits above that cluster — hearing (HS = `hearing_c`), speech production (SP = `deapp_c`) and phonological memory (RW = `erbto`) — so they are entered as **tested covariates** (the missing-indicator method, so no child is dropped for a missing trait), asking "does any carry independent word-reading-gain signal net of the language/letter-sound cluster?". The model's own local DAG figure gains the three nodes with dashed under-test `→gain` edges, mirroring NVMA/behaviour.

**Caveat recorded in the model.** Each upstream trait's effect on gain runs mostly _through_ the baseline-skill cluster that is already conditioned on, so a near-zero adjusted coefficient means "no signal beyond the measured skills", not "unrelated to gain". The residual **time-lagged** confounding (the trait's effect on later skill _development_, not just the t1 level) is genuinely the wave-unrolled-DAG question and is deferred to #250. Dev-fit confirms the path: `hearing`, `speech` and `phonological memory` now appear in `predictor_associations.csv` (all small and shrunk toward zero, as predicted), and `erbto_missing` is auto-dropped as constant on the fitted rows.

A supporting pipeline fix: `fit_adjusted` now re-filters the covariate list against the loaded data after the loader drops any constant `_missing` indicator, so the model never requests a coefficient for a term that was never estimated — the same guard added for the gain/level-factor `adjust_for` path in PR 1.

## Opportunistic de-staling

Per the review decision, the same stale `locked DAG` / `notes/202606231600-…` references in the **mechanism** result partial and the **mediation** reports (MED-064, MED-066) were updated to the revised graph in the same pass, even though those families are owned by #258 / #264 — they are one-line citation fixes and leaving them would be inconsistent. The mechanism and mediation _models_ are unchanged; only the DAG citation moved.
