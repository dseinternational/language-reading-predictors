> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# One artefact interface and a fit-level manifest for the statistical pipeline (#394, tranche 1)

**Date:** 2026-08-07. **Issue:** #394 (complete the statistical pipeline family split and artefact lifecycle refactor), implementation-sequence steps 2–3. **Change:** behaviour-preserving.

## What changed

`pipeline.py` wrote every published table with an inline `df.to_csv(os.path.join(ctx.output_dir, ...))` followed by a manual `ctx.tables[...]` registration — 109 write sites and 104 registrations at the time of the #394 review — and guarded optional artefacts with ad-hoc `except Exception` blocks that printed a one-line warning and recorded nothing. Nothing stated which artefacts a fit produced, which optional ones were skipped, or why; the #394 review flagged exactly this ("without a common structured record of skipped artefacts").

This tranche introduces `statistical_models/artifacts.py` and migrates every `pipeline.py` table write through it:

- **`save_table(ctx, name, df, *, filename=None, required_columns=None, index=False, register=True, required=True)`** — one operation that writes the CSV into the fit's output directory, registers the frame on `ctx.tables`, optionally validates required columns (fail-loud, before anything is written), and records the artefact. The historical three-line idiom becomes one line, and the write and the registration can no longer drift apart.
- **`guard_optional(ctx, label, *, filename, kind)`** — the warn-and-continue guard as a context manager: an optional artefact's failure still prints the same `[yellow]{label} skipped: {exc}[/yellow]` warning and never aborts an expensive fit, but the failure type and message are now persisted instead of scrolling away. Only `Exception` is caught, exactly as the guards it replaces.
- **`write_manifest(ctx)`** — called by the shared `SharedFitStages.finalize_report` after the report-template copy and before publication, so every family gets `artifact_manifest.json` in its output directory with no per-family wiring. The manifest reconciles the fit's artefact log with a recursive directory scan: recorded artefacts carry status (`written` / `skipped`), shape (`n_rows`, `columns`) and any skip reason (`error_type`, `error`); files present on disk but not routed through the interface (figures from plot helpers, shared-writer diagnostics, `trace.nc`, the copied report template) appear as `untracked` with a kind inferred from the extension; a recorded write whose file has vanished is surfaced as `missing` rather than silently dropped.

`StatisticalFitContext` gains an `artifacts: ArtifactLog` field (default-constructed, so no call-site changes), and the interface duck-types the context — `getattr` fallbacks for `tables` / `artifacts` — so the lightweight namespace contexts used by sweep runners and tests keep working.

## Why the manifest scans the directory

Adoption is incremental (this is #394's point: extract helpers first, then move families). If the manifest listed only interface-routed artefacts it would be an inventory of the migration, not of the fit. The reconciliation scan makes it a complete inventory of the fit directory from the first adoption, and the recorded/untracked split doubles as the adoption metric for later tranches. On the four dev verification fits the untracked set is figures and their figure-data CSVs, `trace.nc`, the JSON records and `diagnostics.csv`/`psense_summary.csv` written by the diagnostics module, the report-template copy — and the tables of the not-yet-migrated writers (`itt.py`'s analysis audits, `predicted_scores.py`), which is precisely the adoption gap the split is meant to show (on `lrp-rli-itt-009`: 14 recorded, 12 untracked CSVs).

## What deliberately did not change

- **Behaviour of every existing write.** Filenames, `index` settings, registration keys, registration presence (the handful of deliberately unregistered writes — `analysis_rows.csv`, `proportion_at_zero_ppc.csv`, `readiness_threshold.csv`, the historical-joint per-measure loop — pass `register=False`), warning texts, and which failures are fatal. Two `prior_pushforward` guards keep their produce-only `try` (a save failure there was fatal before and stays fatal); only guards that already wrapped the save were converted to `guard_optional`.
- **The figure emitters' guards.** The ~15 figure guards print `… failed: {exc}` rather than `… skipped: {exc}` and live around plot construction; converting them (and threading their skip records into the manifest) is the natural next slice of #394 step 3, kept out of this PR so the diff stays single-purpose. Their outputs already appear in the manifest via the scan.
- **Writes in other modules.** `itt.py` (4), `diagnostics.py` (3, partly via shared `dse_research_utils` writers), `sensitivity.py`, `predicted_scores.py`, `mechanism_items.py`, `influence.py` and `blending_sensitivity.py` keep their own writes for now; they appear as `untracked`/recorded-elsewhere in manifests until their tranche. Post-hoc regeneration scripts (`regenerate_key_findings.py`, sweep attach) modify fit directories after the manifest is written; the manifest is a fit-time record, not a live index.

## Verification

- **Byte-identical artefacts.** The same four dev fits (`lrp-rli-itt-001` ordinary ITT; `lrp-rli-itt-009` the 423-line floor branch; `lrp-rli-gf-001` gain factors; `lrp-rli-did-001` DiD) were run from `main` and from this branch into scratch roots and every CSV compared byte-for-byte (sampler outputs are seeded and deterministic on one machine). The only differences are the new `artifact_manifest.json` and the run-stamped metadata (`config.json` timestamps/durations), which differ between any two runs.
- **12 unit tests** in `tests/statistical_models/test_artifacts.py`: byte-identity with the legacy idiom (default and `index=True`), registration and record fields, `register=False`, fail-loud `required_columns` (nothing written), duck-typed minimal context, guard swallow/warn/record semantics, `KeyboardInterrupt` propagation, success-after-skip supersedes the record, manifest reconciliation (recorded + skipped + untracked + nested paths + self-exclusion + sorted order), vanished-write `missing` surfacing, and the characterisation guard: `pipeline.py` contains no direct `.to_csv(` call, so a regression back to the inline idiom fails the suite.

## Relation to the #394 acceptance criteria

This delivers "every published table is written and registered through one artefact interface" and "required and optional artefacts are declared, validated and recorded in a structured artefact manifest" for the monolith's own writes, and the mechanism the remaining criteria build on. It does not yet move any family out of `pipeline.py` (steps 4–6), extend typed settings (step 4 of the design), or complete the shared lifecycle stages — those are subsequent tranches.
