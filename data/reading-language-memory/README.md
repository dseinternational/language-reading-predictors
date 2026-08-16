> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

<!-- Copyright (c) 2026 Down Syndrome Education International and contributors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Reading, Language, And Memory Data

Prepared participant-level data for the Byrne, MacDonald, and Buckley longitudinal reading, language, and memory study.

## Files

- `reading_language_memory_data_long.csv` - long-format analysis data, 97 children by 5 annual assessment waves (`485 x 32`).
- `reading_language_memory_data_wide.csv` - one row per child (`97 x 54`).
- `source_provenance.json` - checksum-bound, non-identifying audit record linking the prepared files to the versioned SPSS source.

## Coding

- `subject_id` - anonymised participant identifier.
- `time` - annual assessment wave (`1` to `5`). The 2002 paper reports the first three waves; later waves are available for the Down syndrome sample.
- `readgrp` - reading group:
  - `1` = Down syndrome
  - `2` = average reader
  - `3` = reading-matched
- `sex` - prepared as a `0`/`1` variable. Confirm value labels before reporting sex-specific summaries.
- `age` - age in months.

## Measures

- `basread` - British Ability Scales word reading.
- `basspel` - British Ability Scales spelling.
- `bpvs` - British Picture Vocabulary Scale receptive vocabulary.
- `trog` - Test for Reception of Grammar.
- `woco` - Wechsler Objective Reading Dimensions reading comprehension.
- `basdig` - British Ability Scales recall of digits.
- `bassim` - British Ability Scales similarities/verbal reasoning.
- `basnum` - British Ability Scales number skills.
- `basmat` - British Ability Scales matrices/non-verbal reasoning; available from wave 3 onward in the prepared data.

The prepared long file also includes `_next` and `_gain` columns computed within participant from each wave to the next wave.

## Initial Audit Notes

- The prepared files contain 24 Down syndrome, 42 average-reader, and 31 reading-matched children, matching the starting sample reported in Byrne, MacDonald, and Buckley (2002).
- The first three waves reproduce the paper's Table 2 complete-case means when complete cases are selected separately per measure.
- Source provenance was reconciled on 2026-08-16. The prepared 97-participant wide file matches every case and all 52 shared non-identifying fields in the checksum-pinned 97-case SPSS source retained in the `dsegroup/research-data-analysis` Git history. The historical 96-row CSV was a strict subset that omitted one Down-syndrome participant; source-repository PR #13 repaired it to 97 rows, and the repaired derivative now matches all source assessment fields. See `source_provenance.json` and `notes/202608161340-byrne-source-provenance-reconciliation.md`.
- The prepared files do not include the visual recall variables reported in the paper's correlation tables. Those analyses can only be partially reproduced from the current prepared extract unless the missing variables are recovered.
