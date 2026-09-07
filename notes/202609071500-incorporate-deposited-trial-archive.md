<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Incorporating the deposited trial archive into the repository's own sources

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

Date: 2026-09-07 — **Status: DECIDED** (author instruction, 2026-09-07).

## Decision

The deposited RLI trial archive is **committed to this repository** as `data/dse-rli-trial-data-archive.csv`, verbatim and under its upstream filename, and is licensed with the rest of `data/` under CC BY 4.0. The word-reading missingness bundle resolves that file by default, so `--rli-randomised-archive` becomes an override for a local re-check rather than a requirement, and `scripts/import_rli_randomised_archive.py` becomes optional.

This reverses the run-time-supply arrangement recorded in `data/readme.md`, `METHODS.md` and `notes/202607131900-attrition-audit.md`.

## Why the earlier blocker does not apply

The archive was withheld because the ReShare item's item-level licence field is blank while ReShare's terms describe two possible ShareAlike licences for open deposits, so a downstream user could not safely redistribute it under this repository's CC BY 4.0 data licence.

**Down Syndrome Education International deposited the collection and holds the rights in it** (confirmed by Frank Buckley, 2026-09-07). A deposit licence constrains third parties, not the rights holder, so DSE may license its own data as it chooses. The blocker was real for a downstream user and simply never applied to us. The deposit remains at [doi:10.5255/UKDA-SN-852291](https://doi.org/10.5255/UKDA-SN-852291) and the committed copy matches its published SHA-256, so the correspondence stays checkable.

## What the archive is, precisely

It is **not** a superset of `rli_data_wide.csv`, and neither file contains the other: the wide file has 198 columns for the 54 analysed children, the archive 85 columns for all 57 randomised children. The archive uniquely holds three things:

1. **The screening wave** (`*_ts`: age, word reading out of 30, letter sounds out of 32, expressive and receptive vocabulary out of 170). No derived file carries any screening column.
2. **The three children lost to follow-up**, absent from the derived files.
3. **WPPSI-III Object Assembly at t1** (`object_ass_raw_t1`), the second non-verbal subtest reported in the trial. The repository otherwise carries Block Design alone.

## It does not change any estimand

Measured directly from the file: the three excluded children have **75 of the 85 columns entirely empty**, with no measurement at any of the four timepoints. Their non-empty fields are exactly the design variables and the five screening scores. Recovering their screening profiles therefore does not recover their outcomes, and the suite remains an **available-case modified ITT** analysis. This reproduces what `notes/202607131900-attrition-audit.md` already stated; it is recorded here as a measurement rather than a citation because the temptation to read "we now have all 57 children" as "we now have a full-cohort ITT" is obvious and wrong.

## The one convention retired

The previous design deliberately persisted **no source-to-repository subject-ID crosswalk**. Committing the archive lets anyone holding both files match the two row sets on their shared measurements.

Assessment: this retires a belt-and-braces convention rather than disclosing anything. The identifier sets do not overlap (archive labels are 8 hex characters, the derived files' are `ID_` plus 16); both sets are anonymised; both were already public; and the loader's own 71-field reconciliation had already established that the match is one-to-one and computable by anyone with both files. What is preserved is the part that matters: the loader still discards the archive's labels and generates internal row labels only after every allocation, screening and outcome-count assertion passes, so **no crosswalk reaches any fitted artefact**.

## Implementation

- `data/dse-rli-trial-data-archive.csv` committed (14 KB, 57 × 85), digest `7c6cda36…5ae7`, matching the pinned `RLI_ARCHIVE_CSV_SHA256`.
- `itt_missingness.RLI_ARCHIVE_LOCAL_CSV` added; `missingness_source_path` now falls back to it. An explicit path still wins and still fails loudly if absent, so a re-check against a freshly fetched copy cannot silently fall back to the committed one. A missing committed file returns `None`, which the ITT pipeline continues to report as an incomplete release rather than a clean fit.
- The pipeline's operator message no longer tells the reader to supply a flag that is no longer needed.
- Six tests added: the committed file exists and matches its pinned checksum; it loads and reconciles against the wide file for real; default resolution; explicit override; explicit-but-absent still raises; missing deposit resolves to `None`.
- Docs updated: `data/readme.md` (rewritten, with a files table and the three unique contributions), `METHODS.md`, `docs/models/README.md` and the fit script's help text. The three synced agent instruction files needed no change — their "archive" references are to archived `config.json` files and to the 54-child analysis file, both still accurate.
- **`scripts/import_rli_randomised_archive.py` and its test removed** (288 lines). The script existed to install a file the repository now ships, and every ordinary fit resolves the committed copy without it. Nothing unique is lost: the loader checks the committed file against the pinned digest on every use, a test checks it independently, and the source DOI, landing page, ZIP URL, ZIP digest and CSV digest are all still constants in `itt_missingness.py` and are written into each fit's `itt_missingness_provenance.json`. The one fact that lived only in the script — the member path inside the upstream ZIP — is preserved as `RLI_ARCHIVE_ZIP_MEMBER`, and `data/readme.md` now spells out the four-step manual check an auditor would run. Keeping two supported ways to obtain the same file, one of them no longer the normal one, costs more in explanation than the automated fetch was worth. `--rli-randomised-archive` survives as the way to point the bundle at an independently obtained copy.

## Preserving the bytes

`.gitattributes` sets `*.csv text eol=lf`, which would have rewritten this file's 58 CRLF line endings to LF on checkout and changed its digest — silently breaking the pinned checksum for every other machine and every fresh clone. The deposit ships with CRLF, and committing it verbatim is the whole point, so a `data/dse-rli-trial-data-archive.csv -text` rule now exempts it from end-of-line conversion while still letting git show it as a text diff. Verified by hashing the staged blob: `git cat-file blob :data/dse-rli-trial-data-archive.csv` reproduces `7c6cda36…5ae7` with 58 CRLFs and no bare LF.

## Three findings from verifying this work, none caused by it

1. **Seven Quarto render tests could not pass on Windows: they built a POSIX-shaped minimal subprocess environment and omitted every Windows equivalent. Fixed here.** All five in `test_mechanism_report_forms.py` and both end-to-end render tests in `test_report_restructure.py`. Each forwarded `PATH`, `LANG`, `LC_ALL`, `TMPDIR` and `SYSTEMROOT`, and set `HOME`. Three variables were then missing, and each killed the render at a different point before any assertion was reached:

   | Missing        | Failure                                                                                                                                                          |
   | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | `TMP` / `TEMP` | `PermissionDenied: Access is denied. (os error 5): tmpdir` in Quarto's `initSessionTempDir` — `TMPDIR` is unset on this machine while `TMP`/`TEMP` are `D:\temp` |
   | `USERPROFILE`  | `RuntimeError: Could not determine home directory` from `jupyter_core.paths.get_home_dir` — Windows expands `~` from `USERPROFILE`, not the `HOME` the tests set |
   | `COMSPEC`      | `Failed to spawn 'CMD': entity not found`                                                                                                                        |

   The environment is now built from one documented `RENDER_ENV_KEYS` tuple in `test_mechanism_report_forms.py`, imported by the two fixtures in `test_report_restructure.py` that had their own copies, and extended with `TMP`, `TEMP`, `USERPROFILE`, `HOMEDRIVE`, `HOMEPATH`, `APPDATA`, `LOCALAPPDATA`, `COMSPEC`, `PATHEXT` and `WINDIR`. Absent names are skipped, so it is a no-op on Linux and macOS. All fifteen tests in the two modules now pass; before the fix seven of them failed.

   Confirmed pre-existing before fixing: the failures reproduced with every other change here stashed, and the observed count was order-dependent (four of the five mechanism tests passed in one full-suite run while all five failed in isolation), which is what a retried temp-directory probe falling back to the working directory produces. Quarto itself was never at fault — a minimal document with a Python chunk rendered throughout, and so did the whole `docs/report/` book. This mattered enough to fix rather than file: the tests guard the shared report partials the technical report will be built from, and on Windows they were failing for reasons unrelated to anything they assert.

2. **The report book renders only when Quarto is pointed at the project environment.** Without `QUARTO_PYTHON` set to `.venv/Scripts/python.exe` (or the environment activated), `docs/report/` fails with `ModuleNotFoundError: No module named 'dse_research_utils'`. Quarto also warns that `output-dir: ../../output/report` lies outside the project directory. Both belong in the report runbook. With the variable set, all twelve files render.
3. **A mangled `PYTENSOR_FLAGS` creates a stray directory in the repository root, which would mark every fit dirty.** The ambient value on this machine is `base_compiledir=V:\packages\pytensor`; the backslashes are eaten somewhere in the flag parsing, leaving a _relative_ path whose name is `packages` and `pytensor` run together, which PyTensor then creates in the working directory and fills with a numba cache. It is untracked, so `git status --porcelain` is non-empty and every fit in a sweep would record `dirty: true` — the provenance blemish the 2026-09-01 batch went to some trouble to avoid. The fix is to set the variable with forward slashes (`base_compiledir=V:/packages/pytensor`); it is a machine-level setting, so it is reported rather than changed here. Until it is fixed, a sweep should export a corrected value for the run.

## Follow-up worth a decision, not taken here

**Object Assembly is now available and nothing uses it.** The general-ability proxy across the suite is Block Design alone, and `notes/202607172345-design-lessons-for-future-studies.md` names single-indicator constructs and the absence of a real general-ability battery as two of the study's measurement walls. A two-indicator non-verbal composite would not clear that wall, but it would narrow it. Wiring it in is an analytic change — it touches the data schema, the ability covariate, several adjustment sets and the DAG's account of what `GA` is proxied by — so it is left for a separate decision rather than folded into a data-provenance change.

## Related

- `notes/202607131900-attrition-audit.md` — the 57 → 3 → 54 → 53 flow this rests on.
- `notes/202609071300-technical-report-plan-v2.md` — the technical-report plan, whose data chapter and reproducibility appendix this simplifies.
- `METHODS.md`, "Word-reading missing-data sensitivity" — the estimand the archive serves.
