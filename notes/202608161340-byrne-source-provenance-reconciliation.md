> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Byrne/RLM source-provenance reconciliation

## Decision

The prepared 97-participant Byrne reading-language-memory extract is confirmed as a faithful derivative of the authoritative 97-case SPSS source. The historical 96-row CSV was an incomplete later derivative that omitted one Down-syndrome participant; it was not evidence that an unexplained participant had been added to the prepared extract. `dsegroup/research-data-analysis` PR #13 repairs that CSV to 97 rows while retaining all 96 legacy pseudonyms. The repaired CSV and the prepared extract now both match all 97 source assessment rows. The dataset-level source-provenance gate may therefore be marked confirmed. This decision does not resolve provisional instrument ceilings, the `basmat` identity question or the missing visual-recall measures.

## Evidence

The source is the identifying `projects/reading-language-memory/original/data12345.sav` blob retained in the private `dsegroup/research-data-analysis` Git history at revision `f36df93fe946b975cd701867a117e9ac188a1551`. Its Git blob identifier is `591e14ceee2ffe61fe8af8e51ea35ac86ae8436f` and its SHA-256 digest is `e3cac5ff644ab9126fba25803e677f9492e3e076d8d611d8b3c0aa7ea322952c`. It contains 97 cases with the published starting-group counts: 24 children with Down syndrome, 42 average readers and 31 reading-matched children.

The repository's prepared wide file has SHA-256 `b2262d6b3b7102594b3424c4a72f4237dc84087a7b18f6fc815ccdcd0d10a55c`. After excluding source-only identifiers and prepared-only identifiers, all 97 participant rows have unique fingerprints over the same 52 group, age, speed and assessment fields. The source and prepared fingerprint multisets are identical: 97 participants match and there are zero value differences. The 485-row long file has SHA-256 `68ea2e9c847c908b7217431af76abd45a940099ced2bfd9acf4dd69ba7e2e5f6`; every child has five rows and all 485 rows reproduce the corresponding wide-file group, sex and wave-specific source values without a difference.

The SPSS blob does not contain the prepared `sex` column. The preparation script injected its 97 binary values from a separate hard-coded list, so the source comparison cannot independently validate either those values or their labels. The source repair retains the 96 legacy CSV sex values and uses the prepared value, recoded from 0/1 to the CSV's 1/2 convention, for the recovered row. Its audit also records one disagreement between the legacy and prepared sex values among the 96 matched records. No currently registered RLM statistical model uses `sex`; the data README's instruction to confirm its value labels before any sex-specific analysis remains in force. This limitation is separate from the participant-count and assessment-lineage discrepancy resolved here.

The historical CSV was pinned at revision `fab4e8f0b513cd2f275ae1a29bed4c695d7f1ef6`, Git blob `a7a7c2a3ca97c8a5caf591e7beae349492c72129`, SHA-256 `e36e0e2dd880031870d57dd7e2620a27c9cc9c67ee58760f50285725d756997e`. Its 96 participant fingerprints are all present in the SPSS source and prepared extract; it has no contradictory or additional row. It contains 23 rather than 24 children with Down syndrome and omits prepared identifier `ID_25873B41B04B6AE6`, whose recorded assessments cover waves 1–3. The apparent discrepancy was therefore one omitted source row in this derivative CSV.

The repaired CSV is pinned at revision `79ceb55086d502a739b2fd153990916c58096a25`, Git blob `2cac37086cd299cf958ff51c2554faa54b3e7c70`, SHA-256 `a9098f69e6854d31c665c7f2544ce8f7a26a4417606590bb00c0f18eed96ee6d`. It preserves the 96 legacy codes and assessment rows, restores the omitted Down-syndrome participant under a collision-checked 12-character alias derived from the current HMAC-based prepared identifier, and exactly matches all 97 source rows across the 52 shared assessment fields.

## Reproduction and privacy

`scripts/audit_rlm_source_provenance.py` reproduces the comparison from the checksum-pinned SPSS, historical CSV and repaired CSV Git blobs and verifies the committed `data/reading-language-memory/source_provenance.json` manifest. Run `python scripts/audit_rlm_source_provenance.py --source-repository /path/to/research-data-analysis`. The script requires authorised access to the source repository and reads the identifying SPSS blob only through a temporary file. It compares non-identifying fields and prints no names or source codes. The identifying source must not be copied into this repository.

## Release consequence

Future RLM fits snapshot `source_provenance_confirmed = true` and the manifest path in their publication-input contract. Previously stored fits correctly retain the earlier unresolved input snapshot and do not become publishable merely because the catalogue changed; they must be regenerated. Fits using `basspel`, `woco` or `basnum` remain blocked by provisional bounded-count denominators, and fits using `basmat` remain blocked by unresolved instrument identity.

## Reference

Byrne, A., MacDonald, J. and Buckley, S. (2002). Reading, language and memory skills: A comparative longitudinal study of children with Down syndrome and their mainstream peers. _British Journal of Educational Psychology_, 72(4), 513–529. https://doi.org/10.1348/00070990260377497
