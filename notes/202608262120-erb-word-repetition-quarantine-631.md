# ERB word-repetition anomaly: quarantine to missing pending source verification (#631 finding 3)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

Date: 2026-08-26. Scope: the ERB word/nonword repetition columns (`erbword`, `erbnw`, `erbto`) in `data/rli_data_long.csv` and their derived `_next`/`_gain` columns; the `data_utils.load_data()` load path.

## The anomaly

The t4 row for subject `ID_FDCBDCF29AC0BF03` (line 217 of the archived CSV; waitlist arm, `group == 2`) records `erbword = 28`, `erbnw = 14`, `erbto = 14`. The ERB total is by construction the sum of its word and nonword parts, and the additivity identity `erbto == erbword + erbnw` holds exactly on 201 of the 202 rows where all three values are present — this cell is the sole violation. It is also an outlier on its own terms: `erbword`'s maximum on every other row in the archive is 18 (`erbnw` likewise tops out at 18, `erbto` at 36), so 28 exceeds anything the measure produces elsewhere. The corruption propagates into the same child's t3 derived columns: `erbword_next = 28`, `erbword_gain = 12`, `erbto_next = 14`, `erbto_gain = -16`.

A word/total transposition — the true values being `erbword = 14`, `erbto = 28` — is the obvious reading: it would satisfy the identity, respect both observed maxima, and turn the implausible `erbto_gain = -16` into a plausible small change (`28 - 30 = -2`). It is nevertheless a **hypothesis, and it has explicitly not been applied**. The source archive cannot be verified from this checkout, and repairing archived data by inference is exactly the kind of silent judgement call the #631 decision rule forbids.

## The decision

Per #631 finding 3, the affected cells are **quarantined to missing** pending verification against the source archive: `erbword` and `erbto` at t4 for this child, and the dependent derived cells `erbword_next`, `erbword_gain`, `erbto_next`, `erbto_gain` at t3. `erbnw = 14` is retained — it is consistent under both the recorded and the transposed reading. The archived CSV itself is untouched (archived data are never edited); the quarantine is applied at load.

This supersedes, by reference, the "unresolved" status the value carries in the 2026-07-14 design note (`notes/202607141030-time-lagged-model-designs.md`, the `W → RW` deferral discussion): the value is now handled — recorded as missing with a documented provenance trail — though still not _resolved_, which requires the source archive. That note is deliberately not retro-edited.

## What the loader now enforces

`data_utils.load_data()` (the gradient-boosting load path) now:

1. Applies the module-level `KNOWN_BAD_CELLS` quarantine before `configure_data_types`, setting the six cells above to `pd.NA` and emitting a single warning that names this note.
2. Runs `validate_erb_consistency()` after the quarantine: on every row where all three ERB columns are present it requires `erbword + erbnw == erbto`, all three values non-negative, and the observed-maximum soft caps `erbword <= 18`, `erbnw <= 18`, `erbto <= 36` (the caps are observed maxima pending documentation of the ERB test ceilings — the ERB columns have no registered `Measure` and no confirmed denominator). Any **new** violation raises `ValueError` naming the subject and time; `KNOWN_BAD_CELLS` is the single sanctioned bypass.

The statistical-model loaders in `statistical_models/preprocessing.py` are routed through it too: every one of them now reads the archive through the shared `read_source_csv()`, which applies the same quarantine and the same consistency validation before the frame reaches any family's own missing-data contract. A cell quarantined for the gradient-boosting path is therefore quarantined for the Bayesian fits as well — without that, the corrupt `erbto` still reached every fit adjusting for it, and `LRP-RLI-PL-005`, where it is the exposure itself. The archived CSV is never modified; the quarantine applies to the loaded frame only.

## Affected results

Everything below consumed the corrupt values before this change and needs refitting (or at minimum re-checking) once the source archive has been verified — whether that verification confirms the transposition, some other correction, or genuine missingness:

- **Gradient-boosting models (all)**: `erbword`, `erbnw` and `erbto` (and their `_gain`/`_next` derivatives) sit in the default predictor sets, so the corrupt t4/t3 cells entered every GB fit's predictor matrix. Fits through `load_data()` now see the quarantined cells as missing (LightGBM handles NaN natively), so post-change GB refits will differ marginally from stored ones.
- **GB models with ERB targets**: LRP-RLI-GBL-018 / LRP-RLI-GBG-018 and LRP-RLI-GBL-019 / LRP-RLI-GBG-019 model the ERB measures directly, so the corrupt cells were target rows, not just predictor cells.
- **Bayesian fits adjusting for `erbto` at t4** (the `rw` phonological-memory adjuster; stored fits carry the corrupt `erbto = 14`, and refits now see it as missing): the `gain_factors` suite LRP-RLI-GF-001–013 (period-3 rows, where t4 is the post wave), the `concurrent` family LRP-RLI-CA-001–009 and CA-307, the `block_exposure` family LRP-RLI-BX-001–004 and BX-103, the adjusted LRP-RLI-ADJ-065, and the `lcsm` / `growth` / `long_corr_factor` wave panels wherever `erbto` enters as a per-wave covariate.
- **`pooled_levels` PL-005** (and its PL-006 comparator), where `erbto` is the exposure itself rather than an adjuster.

The distortion is one cell in ~200, entering mostly as a mean-filled standardised covariate, so no stored headline is expected to move materially — but none of these fits should be republished on the stored artefacts once the cell's true value is known.

## Follow-up

Verify the cell against the source ERB record for `ID_FDCBDCF29AC0BF03` at t4. If the transposition is confirmed, correct the archived CSV in a reviewed data-change commit, remove the corresponding `KNOWN_BAD_CELLS` entries (the validator then guards the corrected values), and refit the affected models. If the record is unrecoverable, the quarantine becomes the permanent handling and only the refit step remains.
