> [!NOTE]
> Provenance and external-archive handling drafted by a LLM-based AI tool (Codex/GPT-5).
>
> Deposited-archive incorporation drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Data

## Files

| File                               | Rows                       | What it is                                                                                 |
| ---------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------ |
| `rli_data_long.csv`                | 216 (54 children × 4 waves) | The analysis file: one row per child per assessment wave, with derived `_gain` and `_next` columns. |
| `rli_data_wide.csv`                | 54                         | One row per child.                                                                          |
| `dse-rli-trial-data-archive.csv`   | 57                         | The deposited trial archive, byte-for-byte as published (see below).                        |
| `reading-language-memory/`         | 97 children × 5 waves      | The separate Byrne, MacDonald and Buckley cohort — see its own `README.md`.                 |

## The deposited trial archive

`dse-rli-trial-data-archive.csv` is the file Down Syndrome Education International deposited as the open-access UK Data Service collection [Reading and language intervention for children with Down syndrome: Experimental data](https://doi.org/10.5255/UKDA-SN-852291). DSE deposited the collection and holds the rights in it, so it is distributed here under this repository's own CC BY 4.0 data licence. It is committed verbatim, under its upstream filename and matching the upstream SHA-256, so the correspondence to the published deposit stays checkable; that is why its name departs from the snake_case convention of the derived files beside it.

It is **not** a superset of `rli_data_wide.csv`, and neither file contains the other. The wide file carries 198 columns for the 54 analysed children, including speech, repetition, language-sample, socio-economic, behaviour and attendance measures the archive omits. The archive carries 85 columns for all 57 randomised children, and uniquely holds:

- **the screening wave** (`*_ts` columns: age, word reading out of 30, letter sounds out of 32, and expressive and receptive vocabulary out of 170), which appears nowhere in the derived files;
- **the three children lost to follow-up** after randomisation, who are absent from the derived files;
- **WPPSI-III Object Assembly at t1** (`object_ass_raw_t1`), the second non-verbal subtest reported in the original trial, where the derived files carry Block Design alone.

The three lost children have screening records only: 75 of the 85 columns are empty for them, with no measurement at any of the four timepoints. Recovering their screening profiles therefore does **not** recover their outcomes, and does not turn any analysis into a full-cohort intention-to-treat estimate. The suite remains an available-case modified ITT analysis; see `METHODS.md`.

**What consumes it.** The mandatory word-reading missing-data sensitivity for `lrp-rli-itt-010`, and nothing else. The loader resolves this committed file by default, so a clean checkout runs that bundle with no extra step. `scripts/fit_statistical_model.py … --rli-randomised-archive PATH` points the same bundle at an independently obtained copy instead, which is not needed for an ordinary fit.

**Verifying this copy against the deposit.** Fetch `https://reshare.ukdataservice.ac.uk/852291/1/DSE_Data.zip`, check it against the pinned ZIP digest, extract the member `DSE_Data/dse-rli-trial-data-archive.csv` and check that against the pinned CSV digest. All four values are recorded as constants in `statistical_models/itt_missingness.py`, and each fit repeats the source DOI, landing page, ZIP URL, ZIP digest and CSV digest in its own `itt_missingness_provenance.json`.

**What the loader verifies**, on every use: the pinned upstream checksum; the published flow of 57 randomised children (29 intervention and 28 waiting control) to 54 analysed (28 and 26); the archive-derived 53 observed t2 word-reading outcomes (28 and 25); and a one-to-one 71-field reconciliation of the 54 included rows against `rli_data_wide.csv`.

**Identifiers.** The archive carries its own published anonymised `subject_id` labels, which do not overlap the derived files' internal `ID_*` labels. The loader discards the archive labels and generates internal row labels only after every assertion passes, so no subject-ID crosswalk is written into any fitted artefact. Committing the archive does let anyone holding both files match the two row sets on their shared measurements, which the 71-field reconciliation already established was possible; both sets are anonymised and both were already public, so this retires a belt-and-braces convention rather than disclosing anything new.

## License

The data committed to this directory, including the deposited trial archive, are licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) — see `LICENSE` for details.
