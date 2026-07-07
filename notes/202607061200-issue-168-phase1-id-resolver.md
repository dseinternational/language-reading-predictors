# Model-ID migration — Phase 1: resolver + aliases + metadata (#168)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

Delivers **Phase 1** of the model-ID scheme migration planned in
`notes/202607021145-issue-168-model-id-migration-plan.md`: the **non-destructive
compatibility layer**. Nothing is renamed — module files, `docs/models/` folders,
output directories and report titles all keep their legacy ids. The canonical
scheme now exists behind a resolver, both fit CLIs accept either id form, and every
model carries its canonical id as read-only metadata. Phase 2 (the mechanical
rename) can then land incrementally behind this layer.

## Canonical scheme

`<project>-<study>-<family>-<nnn>`, three forms (numbers zero-padded to 3;
the legacy ids use 2, which the resolver accounts for):

| Use            | Form              | Example           |
| -------------- | ----------------- | ----------------- |
| Display / docs | UPPER, hyphen     | `LRP-RLI-ITT-010` |
| CLI / paths    | lower, hyphen     | `lrp-rli-itt-010` |
| Python module  | lower, underscore | `lrp_rli_itt_010` |

Family codes (by `ModelSpec.kind`, or the `gbg`/`gbl` prefix for GB models):
`itt`/`joint → ITT`, `gain_factors → GF`, `level_factors → LF`, `aligned → AL`,
`did → DID`, `horseshoe → HS`, `corr_factor → MM`, `mechanism → MECH`,
`mediation`/`mediation_multi → MED`, `adjusted → ADJ`, `lcsm → LCSM`,
`dose_response → DOSE`, GB gain → `GBG`, GB level → `GBL`. Two families that
post-date the plan note are added here: `growth → GC` (LRP69/70) and
`historical_growth → HG` (the RLM cohort's `rlmhg`). (Family-code spellings
`MECH`/`LCSM`/`DOSE` were confirmed by the research lead on #168, 2026-07-06.)
Variant suffixes follow the controlled vocabulary — `b` companion, `base`
comparator, `d` dose-sensitivity, `a` alternate (e.g. `lrpgf01b → lrp-rli-gf-001b`,
`lrp77base → lrp-rli-dose-077-base`).

## Delivered

- **`src/language_reading_predictors/model_ids.py`** — a dependency-free resolver:
  `parse_legacy` / `to_canonical` / `parse_canonical` / `to_legacy` (pure,
  reversible, three-form), plus `resolve_to_legacy` for the CLIs. The family of a
  _bare_ `lrp##` model (e.g. `lrp65`) is supplied by its caller's `kind`; ids that
  embed their family need none.
- **Metadata accessors** on `ModelSpec` (context.py) and the GB `ModelConfig`
  (common.py): `canonical_model_id` / `legacy_model_id` / `project_code` /
  `study_code` / `family_code` / `variant_role` / `parent_model_id`. Derived on
  read — non-destructive, and an id the resolver cannot parse yields `None` rather
  than breaking a fit. `write_run_metadata` now records the canonical + legacy ids.
- **Dual-ID CLIs** — `fit_statistical_model.py` and `fit_model.py` accept either a
  legacy id (`lrpitt10`) or a canonical id (`lrp-rli-itt-010`, any case/form);
  registry and output lookup stay keyed by the legacy id.
- **Tests** (`tests/test_model_ids.py`) — three-form round-trip for a
  representative id per family + each variant kind; `resolve_to_legacy` for every
  canonical form; and an **import-free whole-inventory sweep** (all 139 model
  modules) asserting every id resolves + round-trips with **no duplicate canonical
  ids**. A guard test fails if a new `kind` is added without a family code.

## Verified

- `ruff check src/ scripts/`, the touched-module test sweep (`test_model_ids`,
  `test_model_definitions`, `test_registry`, `test_reporting`, `test_models`), and
  `npm run format:check` / `npm run spellcheck` all green.
- End-to-end: `python scripts/fit_statistical_model.py lrp-rli-itt-010 --config dev`
  resolves to `lrpitt10`, fits, and writes `config.json` with
  `canonical_model_id = lrp-rli-itt-010`, `family_code = ITT`, `study_code = RLI`.

## Not in this phase (Phase 2)

Renaming module files (`lrpXXX.py → lrp_rli_fam_nnn.py`), `docs/models/{id}/`
folders, report titles + cross-links, script id literals, and deciding the old
`output/` directory disposition — all of which can now land incrementally behind
this resolver, reusing the #116 Phase-A rename mechanics.

## Team sign-off (resolved on #168, 2026-07-06)

The research lead confirmed all five items:

1. Family-code spellings — **`MCH → MECH`, `LCS → LCSM`, `DR → DOSE`**; `GC`/`HG`
   confirmed. Applied here.
2. Variant scheme — **adopt `parent+100` renumbering** (e.g. `lrpgf01b →
lrp-rli-gf-101`), not the controlled-suffix form shipped in Phase 1. **Not yet
   implemented** — it drops the suffix, so (a) it needs a numbering rule for the one
   parent with two variants (`lrp77` has `77a` + `77base`, both → `dose-177`), still
   open for the lead, and (b) the reverse (canonical → legacy) becomes a lookup
   rather than a pure transform. Deferred to the Phase-2 change.
3. Joint ITT models (`lrpitt12/15/15b`) — kept in the `ITT` family. ✓
4. RLM study/family codes — `RLM` + `HG` confirmed; `rlmhg01 → lrp-rlm-hg-001`. ✓
5. Old-output-dir disposition — leave `output/models/{legacy}/` historical. ✓
