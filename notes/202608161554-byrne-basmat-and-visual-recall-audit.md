<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- cspell:ignore basmat BASMAT Byrne MacDonald Buckley readgrp SPSS -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Byrne `basmat` identity and retained visual-recall audit

**Status: measurement-source decision for #338, 2026-08-16.** This audit separates two questions that had previously been conflated: what the retained cohort source calls `basmat`, and whether the published immediate and delayed visual-recall scores survive in that source.

## Evidence boundary

The checksum-pinned original SPSS file contains exactly 54 fields: two identifier fields and the 52 non-identifying fields already reconciled against the prepared extract. Its native assessment inventory includes `BASMAT3`, `BASMAT4` and `BASMAT5`; it contains no additional assessment fields from which immediate or delayed visual recall could be recovered. The source repository independently defines `basmat` as British Ability Scales Matrices/non-verbal reasoning, and the preparation path carries those source fields into the wide and long extracts without renaming the construct.

The cohort's three-wave papers provide an important negative boundary. The baseline report names a 97-child cohort with the same 24 Down-syndrome, 42 average-reader and 31 reading-matched starting groups, and explicitly reports BAS immediate and delayed visual recall (Byrne et al., 1995, DOI [10.3104/reports.51](https://doi.org/10.3104/reports.51)). The final three-wave paper again names the two visual-recall tests but does not list or report a matrices test (Byrne, MacDonald and Buckley, 2002, DOI [10.1348/00070990260377497](https://doi.org/10.1348/00070990260377497)). It therefore confirms that visual recall was collected and that `basmat` is not part of the paper-compatible reported battery; it does not identify the later-wave `BASMAT3`-`BASMAT5` source fields.

The Raven ambiguity came from a different study. Laws et al. (1995) used Raven's Coloured Progressive Matrices in a 14-child follow-up drawn from a separate 51-child memory-strategy project begun in 1991, not the Byrne 97-child mainstream-school cohort recruited in 1993-94 (DOI [10.3104/reports.52](https://doi.org/10.3104/reports.52)). That paper cannot identify the Byrne source's `BASMAT` fields and is not evidence that they contain Raven scores.

## Decisions

1. Confirm `basmat` as the source-native BAS Matrices measure. The existing 28-item denominator remains separately supported by the instrument evidence recorded in `notes/202607161200-byrne-phase-a-window-and-ceilings.md`. This is a source-record decision, not a claim that the 2002 article reports matrices.
2. Treat `basmat` as a later-wave extension only. Its absence from the published three-wave measurement account must remain explicit whenever it is analysed.
3. Accept that immediate and delayed visual recall are absent from every retained field in the checksum-pinned source and its prepared derivatives. The repository cannot reproduce the paper's full memory battery or its full correlation tables. This establishes archival absence, not that the measures were never collected; recovery would require a different original archive.
4. Fresh fits may record `basmat.instrument_identity_confirmed=True`. Stored fits retain their fit-time input contract and do not become publishable retrospectively. `lrp-rlm-hg-009` therefore needs a fresh reporting fit before interpretation; models that also use `basspel`, `woco` or `basnum` remain blocked by their provisional denominators.

## Reproducibility record

`scripts/audit_rlm_source_provenance.py` now validates the exact 52-field analytical inventory, verifies that the only SPSS fields outside it are the two excluded identifiers, records the native `basmat3`-`basmat5` fields and records zero retained visual-recall fields in `data/reading-language-memory/source_provenance.json`. No identifying source values are copied or printed.
