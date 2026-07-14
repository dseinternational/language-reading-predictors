<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

# Flagship coverage: dose-response for L/B + horseshoe ranking for L (#228 items 2 & 3)

Date: 2026-07-14

Related: #228 (suite-gap analysis, items 2 & 3), #116 (horseshoe ranking cross-check), #104/#135/#269 (dose-response family).

## What this adds

Two coverage gaps on the **flagship code outcomes**, both closed by reusing existing machinery (no new factory, pipeline, or family):

- **Dose-response for letter sounds and blending (#228 item 2).** The `dose_response` family covered word reading only (`dose-077`/`177`/`277`), yet L and B carry the two _largest_ ITT effects. Adds `lrp-rli-dose-083` (L, `n_trials=32`) and `lrp-rli-dose-084` (B, `n_trials=10`), each the period-resolved intervention-dose → outcome model, identical in estimand and causal structure to `dose-077` (observational IS→outcome, adjust {G, own-baseline, A}; cumulative dose is the `IS`-collider descendant and is not conditioned on, #269). Adjusted associations, not "dose drives gains" — the no-`ability→dose`-edge assumption (Frank 2012) is the live confounder.
- **Horseshoe ranking cross-check for letter sounds (#228 item 3).** `hs-001`/`002` ranked predictors of word-reading gain/level only; the flagship code outcome had no ranking cross-check against the gradient-boosting layer. Adds `lrp-rli-hs-003` (L gain, cross-checked vs GB `gbg-009`) and `lrp-rli-hs-004` (L level, vs `gbl-009`), mirroring `hs-001`/`002` with the predictor set swapped (L becomes the outcome; word reading `W` enters as a predictor). Ranking cross-checks, not causal.

## Ids

Dose uses the family's bare-legacy numbering (`lrp83`/`lrp84` → `dose-083`/`dose-084`; 78–82 were taken, so 83/84 are the next free bare numbers — non-contiguous within a family is normal here, as with mediation). Horseshoe uses the embedded scheme (`lrphs03`/`lrphs04` → `hs-003`/`hs-004`). No `model_ids` change needed for either.

## Verification

- `ruff check src/ scripts/`, `npm run format:check`, `npm run spellcheck` — clean.
- `pytest tests/test_model_definitions.py tests/test_model_ids.py` — pass (registry ↔ modules ↔ SPEC drift guard).
- Dev-tier fits for all four build, sample and summarise (0 divergences); the dose models show sensible period-resolved dose slopes (e.g. dose-083 L period-1 slope ≈ +0.23). Reporting-tier fits + render fold into the next full-suite refit.

This is preliminary, exploratory work in progress.
