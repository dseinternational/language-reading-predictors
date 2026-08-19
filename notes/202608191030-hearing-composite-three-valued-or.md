> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Hearing composite: derive it with a three-valued OR (one child was coded unknown, then filled to clear)

**Date:** 2026-08-19. **Status:** loader corrected; stored fits not yet refitted (held until the current review of the August 2026 findings is complete).

## What the hearing term is

Every statistical model that adjusts for hearing uses the composite `hearing_c` — 1 when the child has impaired hearing **or** a history of repeated ear infections, 0 when both are clear — through the missing-indicator covariates `hs` (composite, unknown filled to the clear reference 0) and `hs_missing` (1 when unknown), derived by `statistical_models.preprocessing.add_hearing_status` (#244 team decision; `notes/202607101100-dag-revision-team-decisions.md`). The gradient-boosting models do not use the composite: `hearing_c` is in `Variables.DEFAULT_EXCLUDED`, and they take `hearing` and `earinf` separately with LightGBM's native missing-value handling.

## The fault

The stored `hearing_c` column in `data/rli_data_long.csv` and `data/rli_data_wide.csv` was derived upstream (before this repository) with a **strict both-known OR**: the composite is unknown whenever either component is unrecorded. Cross-tabulating the components against the composite at timepoint 1 (54 children):

| `hearing` | `earinf` | stored `hearing_c` | children | correct composite            |
| --------- | -------- | ------------------ | -------- | ---------------------------- |
| 0         | 0        | 0                  | 20       | 0                            |
| 0         | 1        | 1                  | 3        | 1                            |
| 1         | 0        | 1                  | 11       | 1                            |
| 1         | 1        | 1                  | 10       | 1                            |
| 1         | unknown  | **unknown**        | **1**    | **1** — impaired is impaired |
| 0         | unknown  | unknown            | 2        | unknown (0 or 1)             |
| unknown   | unknown  | unknown            | 7        | unknown                      |

A child recorded as hearing-impaired satisfies "impaired hearing or repeated ear infections" whatever their ear-infection record says. The strict OR left that one child unknown, and the missing-indicator fill then coded a **known impairment as the clear reference** (`hs = 0, hs_missing = 1`). The two children with clear hearing and no ear-infection record are genuinely indeterminate and stay unknown. The correct split is therefore **25 flagged, 20 clear, 9 unknown**, not 24 / 20 / 10.

## The fix

`add_hearing_status` now derives the composite from `hearing` and `earinf` with the three-valued rule — 1 if either component is 1; 0 if both are 0; otherwise unknown, with the stored composite used only as a fallback in that unknown case (so an extract that populated `hearing_c` from another record keeps that information), and a `ValueError` if a stored composite contradicts both-known components (`preprocessing.derive_hearing_composite`). The stored `hearing_c` column is left untouched: the checked-in wide file's SHA-256 is pinned for the ITT-010 missing-data sensitivity (`itt_missingness.RLI_LOCAL_WIDE_SHA256`), and the composite is not one of the 71 reconciled archive fields, so the right place for the rule is the loader. `tests/statistical_models/test_preprocessing.py` pins the rule, the one-child case, the inconsistency guard and the 25 / 20 / 9 split on the real file.

## What is affected, and what has not been done

108 stored reporting fits carry `hs` (mechanism 40, gain_factors 18, mediation 17, concurrent 11, level_factors 6, block_exposure 5, pooled_levels 3, lcsm 3, corr_factor 2, joint_mechanism 2, adjusted 1). All of them used the old coding; none has been refitted. The child concerned is in the immediate arm and read 4, 4, 8 and 14 words across the four timepoints — moving one such child from the unknown group (n = 10) into the flagged group (n = 24) cannot manufacture the positive hearing–gain association reported in `notes/202608182200-findings-by-question.md` (question 7) and is unlikely to move any estimate materially, but it is a misclassification on every adjusted model, and the hearing figures in the notes are read as "stored fits, old coding" until the refit. The findings note and the `adjusted` family note (which reports the same +2.4-words figure) carry a caveat to that effect; the other family notes describe hearing only as an adjuster and need no change.

**Refit:** held at the author's request until the review is complete. When it runs, every fit that carries `hs` should be refitted together (the `mechanism` sweep is the bulk of it) and the hearing figures in question 7 re-read from the new artefacts.

## Related wording correction

The same check found that question 8 of the findings note suggested "using the continuous hearing information". There is none — both components are yes/no — so the suggestion now reads "use the two hearing indicators separately".
