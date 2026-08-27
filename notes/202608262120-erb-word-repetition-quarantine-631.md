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
- **Bayesian fits whose prepared analysis frame changes** (the `rw` phonological-memory adjuster and the `erbto` exposure; stored fits carry the corrupt values, and refits now see them as missing). The list is enumerated below rather than reasoned about — see the 2026-08-27 correction.

The distortion is one cell in ~200, entering mostly as a mean-filled standardised covariate, so no stored headline is expected to move materially — but none of these fits should be republished on the stored artefacts once the cell's true value is known.

### Correction, 2026-08-27 — the Bayesian list, measured

> [!NOTE]
> Added by a LLM-based AI tool (Claude Code/Opus 5).

The original bullet named affected Bayesian families by reading the declared adjustment sets. That is wrong in both directions, because a declared confounder only bites when the affected wave is inside the model's window _and_ the affected child is inside its fitted rows. The list below is measured instead: every registered model's plan was resolved and its analysis frame prepared twice — once with `KNOWN_BAD_CELLS` applied and once with the quarantine and its validator neutralised — and the prepared arrays compared.

**Changed (45 RLI models).** Their stored posteriors are not reproducible from current code:

| Family           | Models                                                               |
| ---------------- | -------------------------------------------------------------------- |
| `block_exposure` | BX-001–004, BX-103                                                   |
| `concurrent`     | CA-001–009, CA-307                                                   |
| `gain_factors`   | GF-005, GF-105, GF-205                                               |
| `level_factors`  | LF-002, 003, 006, 009, 010, 106, 202, 203, 206, 209, 210             |
| `mechanism`      | MECH-056, 057, 088, 089, 090, 102, 104, 156, 157, 188, 189, 190, 204 |
| `pooled_levels`  | PL-003, PL-004, PL-005                                               |

**Where the original bullet was wrong.** It claimed the whole `gain_factors` suite GF-001–013; only the three off-floor `P` models change, and GF-001–004 and 006–013 resolve no ERB covariate at all. It omitted `level_factors` (eleven fits) and `mechanism` (thirteen) entirely, and both are affected. It named `growth` and `long_corr_factor`, neither of which changes. `PL-006` does not change while `PL-003`/`PL-004` do. Two direct refits confirm the negative side: `med-059` and `med-060` — the latter carrying `erbto` in its effective confounder set — both reproduce their pre-quarantine posteriors byte-for-byte, with unchanged fitted-row digests.

**Not measured (16 RLI models).** Their loaders take arguments the harness did not supply, so they are neither confirmed changed nor confirmed clean: ADJ-065, LCSM-067/081/082/091/181, MED-076/092/176/276, MM-001/002/101/102, SURV-009/011.

**Out of scope.** All 25 RLM (Byrne) fits read a different archive, which this quarantine does not touch.

Two of the changed fits, `CA-007` and `CA-307`, are the two halves of a released phoneme-blending link pair, so the pair's published numbers rest on pre-quarantine data on both sides.

The probe is `notes/assets/202608271400-erb-quarantine-impact.py`; re-run it after any loader change. The refits are tracked separately rather than folded into the 2026-08-27 closing pass — see `notes/202608271200-closing-584-588-residuals.md`.

## Follow-up

Verify the cell against the source ERB record for `ID_FDCBDCF29AC0BF03` at t4. If the transposition is confirmed, correct the archived CSV in a reviewed data-change commit, remove the corresponding `KNOWN_BAD_CELLS` entries (the validator then guards the corrected values), and refit the affected models. If the record is unrecoverable, the quarantine becomes the permanent handling and only the refit step remains.
