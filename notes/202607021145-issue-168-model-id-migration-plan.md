# Model-ID scheme migration for multi-study support — plan (#168)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

This is a **proposal for sign-off**, not an implementation. It resolves the three
open decisions in #168 (plus a joint-outcome sub-question) against the actual model inventory, fixes the canonical
scheme, and specifies a low-risk two-phase migration. No files are renamed here.

## Why

The current IDs (`lrpitt10`, `lrpgbg12`, `lrp65`) hard-code the project (LRP) and
assume the single Reading & Language Intervention study (RLI). The Reading,
Language & Memory (RLM) study is coming (#164/#165), and its models already need a
home (`rlmhg01`). A scheme that makes **project · study · family · number**
explicit lets studies coexist without collisions and makes cross-references
self-describing.

## Current inventory (the migration's blast radius)

**136 models = 86 Bayesian statistical + 50 gradient-boosting.** Grouped by
`ModelSpec.kind` (statistical) / registry (GB):

| Family (kind)        | Count | Current ids                          | Proposed code |
| -------------------- | ----: | ------------------------------------ | ------------- |
| ITT                  |    23 | `lrpitt01–11,13,14,17–24`            | `ITT`         |
| ITT — joint outcome  |     3 | `lrpitt12,15,15b`                    | `ITT` (§D-4)  |
| gain factors         |    16 | `lrpgf01–08` + `…b` companions       | `GF`          |
| level factors        |     8 | `lrplf01–08`                         | `LF`          |
| aligned per-protocol |     9 | `lrpal01–08` + `lrpal01d`            | `AL`          |
| difference-in-diff   |     8 | `lrpdid01–07` + `lrpdid07base`       | `DID`         |
| horseshoe            |     2 | `lrphs01,02`                         | `HS`          |
| correlated factor    |     1 | `lrpmm01`                            | `MM`          |
| mechanism            |     8 | `lrp56,57,58,71,72,72base,73,73base` | `MCH` ⚠ new   |
| mediation            |     2 | `lrp59,62`                           | `MED` ⚠ new   |
| mediation (multi)    |     1 | `lrp64`                              | `MED` (§D-2)  |
| adjusted association |     1 | `lrp65`                              | `ADJ` ⚠ new   |
| latent change (LCSM) |     1 | `lrp67`                              | `LCS` ⚠ new   |
| dose–response        |     3 | `lrp77,77a,77base`                   | `DR` ⚠ new    |
| GB — gain            |    22 | `lrpgbg01–22`                        | `GBG`         |
| GB — level           |    28 | `lrpgbl01–28`                        | `GBL`         |

The rows marked ⚠ are the **bare `lrp##` models** whose family is invisible in
today's id — the core of open-decision #3. Their family is unambiguous from
`kind`, so codes can be assigned mechanically.

Each id also drives: module filename, registry key / `model_id`, `output/models/{id}/`
and `output/statistical_models/models/{id}-{config}/`, `docs/models/{id}/`, report
titles + cross-links, `scripts/*` id literals, tests, `METHODS.md` / `AGENTS.md` /
`CLAUDE.md` / `.github/copilot-instructions.md`, and `config/spellcheck/allow-en.txt`.

## Canonical scheme (confirmed)

`<project>-<study>-<family>-<nnn>`, three representations:

| Use                | Form              | Example           |
| ------------------ | ----------------- | ----------------- |
| Display / docs     | UPPER, hyphen     | `LRP-RLI-ITT-010` |
| CLI / output paths | lower, hyphen     | `lrp-rli-itt-010` |
| Python module      | lower, underscore | `lrp_rli_itt_010` |

Numbers are zero-padded to 3 digits. `RLI` = Reading & Language Intervention,
`RLM` = Reading, Language & Memory.

## Resolved open decisions

**D-1 — GB family code: use `GBG` / `GBL`** (not a flat `GB` + metadata). The
gain/level split is load-bearing (different predictor frames, established in #116),
and visible codes keep short references legible. Matches the issue author's
preference.

**D-2 — Variants: keep numbers stable, add metadata, use a controlled suffix.**
~18 models are variants (`…b` companions ×11, `…base` comparators ×4, `lrpal01d`
dose-sensitivity, `lrp77a`). Recommendation — the **lowest-risk** path:

- Add `variant_role` + `parent_model_id` metadata (the real fix for machine use).
- Preserve the parent's number and attach a **controlled** suffix from a fixed
  vocabulary — `b` = paired/treated-only companion, `base` = pooled/comparator,
  `d` = dose-sensitivity, `a` = alternate spec — e.g. `lrpgf01b → lrp-rli-gf-001b`,
  `lrpdid07base → lrp-rli-did-007-base`.

This keeps the migration a deterministic, reversible number-preserving map and
still satisfies "variant models have a consistent, documented representation." The
issue's alternative (renumber variants to `parent+100`, e.g. `gf01b → gf-101`)
gives cleaner base IDs but adds a renumbering judgement per variant and breaks the
visual parent link — I recommend **against** it for the first migration; it can be
revisited once the resolver exists.

**D-3 — Non-listed families: covered by the table above** (`MCH`, `MED`, `ADJ`,
`LCS`, `DR` added to the listed `ITT/GF/LF/GBG/GBL/DID/AL/HS/MM`). Exact spellings
(`MCH` vs `MECH`, `LCS` vs `LCSM`) are a cosmetic call for sign-off; the plan works
with any fixed choice.

**D-4 — Joint-outcome ITT models (`lrpitt12/15/15b`): keep them in the `ITT`
family.** They are ITT analyses with a joint (understood + spoken) outcome, so the
analysis family is still ITT; the joint structure is better carried as
`variant_role` metadata than as a separate `JNT` family code. (Listed as sign-off
item 3 for confirmation.)

## Mapping rule (number-preserving)

For every model: `family_code = FAMILY[kind]`, `number = <the digits already in the
id>`, `study = RLI` (all current models), suffix retained from the controlled
vocabulary. Deterministic and reversible — no renumbering:

```
lrpitt10     -> lrp-rli-itt-010          lrp56        -> lrp-rli-mch-056
lrpitt15b    -> lrp-rli-itt-015b         lrp64        -> lrp-rli-med-064
lrpgbg12     -> lrp-rli-gbg-012          lrp65        -> lrp-rli-adj-065
lrpgbl12     -> lrp-rli-gbl-012          lrp67        -> lrp-rli-lcs-067
lrpgf01b     -> lrp-rli-gf-001b          lrp77base    -> lrp-rli-dr-077-base
lrpdid07base -> lrp-rli-did-007-base     rlmhg01      -> lrp-rlm-hg-001   (RLM study, family pending §sign-off 4)
```

Bare `lrp##` numbers (56–77) are kept for provenance even though they are not
per-family-sequential — traceability beats tidy renumbering, and `legacy_model_id`
records the original regardless.

## Migration — two PRs (never a blind search-and-replace)

**Phase 1 — resolver + aliases + metadata (no rename; fully backward-compatible).**

- New `statistical_models/model_ids.py` (+ a GB equivalent or shared): canonical
  parse/validate, the three-form formatters, and legacy⇄canonical maps built from
  the taxonomy above.
- Add `canonical_model_id` / `legacy_model_id` / `project_code` / `study_code` /
  `family_code` / `variant_role` / `parent_model_id` to `ModelSpec` and `ModelConfig`.
- CLI resolution in `fit_model.py` / `fit_statistical_model.py` accepts **both** old
  and new ids; output lookup stays legacy-compatible so existing artefacts read.
- `write_run_metadata` records both ids.
- Tests: id parse/format round-trip, alias resolution both directions, no duplicate
  canonical ids, every discovered model resolves. (Compose with #172's
  `discover_models` once it merges.)

**Phase 2 — mechanical rename (once the map is proven).**

- Rename modules `lrpXXX.py → lrp_rli_fam_nnn.py`; rename `docs/models/{id}/`;
  update report-copy to canonical with legacy fallback during transition.
- Update every id literal in `scripts/*`, tests (keep a few explicit legacy-alias
  tests), `METHODS.md`/`AGENTS.md`/`CLAUDE.md`/copilot-instructions, report titles +
  cross-links, and `allow-en.txt`.
- Decide old `output/` dirs: leave as historical (recommended) vs migrate/symlink.

### Reuse the #116 Phase-A rename mechanics

The GB `lrpNN → lrpgbg/lrpgbl` rename (#150) is the proven template: a **guarded
regex** mapping only the intended id tokens; **read-into-a-variable-then-write**
(the truncate-before-read hazard emptied 42 `.qmd` files once); update package `__init__`

- docs dirs + script literals + tests + `allow-en` together; verify with a repo-wide
  `git grep` for legacy ids returning only intended hits.

## Risks / verification

- **Churn size** — 136 models × ~6 surfaces each. Phase 1's alias layer means Phase 2
  can land incrementally (per-family) behind the resolver without breaking CLIs.
- **RLM study code** — the scheme must name RLM before its models land; `rlmhg01`
  (from #165) is the first and needs its study/family slot fixed here.
- Gates: `ruff check src/`, `npm run format:check`, `npm run spellcheck`, plus the
  Phase-1 id tests, all green before either PR merges.

## Remaining for team sign-off

1. Family-code spellings (`MCH`/`MECH`, `LCS`/`LCSM`, `DR`).
2. Variant scheme: controlled-suffix + metadata (recommended) vs `parent+100` renumber.
3. Joint ITT models (`lrpitt12/15/15b`) — keep in `ITT` (recommended) or split to a `JNT` family.
4. RLM study/family codes (needed for `rlmhg01`).
5. Old-output-dir disposition (leave historical vs migrate).
