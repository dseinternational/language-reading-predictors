<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Analysis-set and outcome-attrition audit: is an informative-missingness sensitivity warranted?

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8). The within-archive counts below were recomputed from `data/rli_data_long.csv` by `scripts/attrition_audit.py`; re-run it to reproduce. Published randomisation counts come from Burgoyne et al. (2012). This note is evidence for the issue #230 §3 decision — it does not itself close the item.

> [!NOTE]
> Substantially revised by a LLM-based AI tool (Codex/GPT-5) following the ITT audit in issue #341.

Date: 2026-07-13 — relates to issue #230 §3 (measurement and missingness debts).

## The question

Issue #230 §3 flags that the suite's complete-case comparators only handle _covariate_ missingness, not _outcome_ dropout, and asks whether a delta-adjustment or other informative-missingness sensitivity is needed. The first version of this audit considered only the 54 children present in the archived CSV. Issue #341 identified the missing outer denominator: Burgoyne et al. randomised 57 children, 29 to immediate intervention and 28 to wait-list control, whereas the repository contains 54. This revision separates absence from the modelling dataset from missing cells within it. Trial allocation and analysis counts are documented in Burgoyne et al. (2012), DOI [10.1111/j.1469-7610.2012.02557.x](https://doi.org/10.1111/j.1469-7610.2012.02557.x).

## The evidence

### Published randomised population versus archived dataset

| Arm                    | Randomised | In archived dataset | Absent from dataset |
| ---------------------- | ---------- | ------------------- | ------------------- |
| Immediate intervention | 29         | 28                  | 1                   |
| Wait-list control      | 28         | 26                  | 2                   |
| **Total**              | **57**     | **54**              | **3**               |

The three absent children cannot appear in a wave-by-wave audit of `rli_data_long.csv`; their outcome availability and reasons for absence are not encoded in this repository. The analyses preserve assigned arms for the 54 archived children but are available-case or modified-ITT analyses, not full-randomised-cohort ITT analyses.

### Missing outcome cells within the 54-child archive

Per outcome, the next table gives children with a non-missing score at each wave and **single-leg** missingness across the transitions used by headline estimands: the randomised **ITT window** (t1 baseline but no t2 post), the **t2→t3 crossover leg** (t2 but no t3), and the **maintenance leg** (t3 but no t4). The columns are per-leg, so the full DiD crossover chain t1→t2→t3 is complete for an outcome iff both its ITT-window and t2→t3 columns are zero. Every count in this table is conditional on inclusion in the 54-child archive.

| Outcome                  | t1  | t2  | t3  | t4  | ITT-window | t2→t3 | t4  |
| ------------------------ | --- | --- | --- | --- | ---------- | ----- | --- |
| W word reading           | 53  | 53  | 53  | 51  | 0          | 0     | 2   |
| R receptive vocabulary   | 54  | 54  | 54  | 53  | 0          | 0     | 1   |
| E expressive vocabulary  | 54  | 54  | 54  | 53  | 0          | 0     | 1   |
| L letter-sound knowledge | 54  | 54  | 54  | 52  | 0          | 0     | 2   |
| B phoneme blending       | 54  | 54  | 54  | 53  | 0          | 0     | 1   |
| F basic concepts (CELF)  | 54  | 54  | 54  | 52  | 0          | 0     | 2   |
| T receptive grammar      | 54  | 54  | 54  | 53  | 0          | 0     | 1   |
| P phonetic spelling      | 54  | 53  | 53  | 54  | 1          | 0     | 0   |
| N nonword reading        | 50  | 53  | 52  | 52  | 0          | 1     | 1   |

## Reading

Within the archived dataset, outcome-cell missingness is small. That narrower fact must not be used to erase the three children absent before this audit begins:

- **The randomised ITT window (t1 → t2) is internally complete** for the seven graded standardised outcomes (W, R, E, L, B, F, T): every archived child with a baseline has a randomised post-score. Word reading has 53 rather than 54 scores at both waves. Phonetic spelling has one child with a t1 score but no t2 score.
- **The DiD crossover chain (t1 → t2 → t3) is complete** for every graded outcome (both the ITT-window and t2→t3 legs are 0). The single t2→t3 gap is one child on the floored nonword-reading measure (N).
- **Only the final maintenance wave (t4) loses anyone**, and at most two children (W, L and F lose 2; the rest lose 0–1). t4 is post-crossover — both arms are treated by then — so it feeds the aligned / level / growth analyses, not the randomised ITT or the DiD contrast.

The nonword t1 count is 50, not 54. Although the regression has no baseline covariate, the floor-transition estimand conditions on an observed t1 zero. Of the 53 children with a t2 score, three have unknown baseline-floor eligibility (one intervention, two control) and are excluded; 36 children are eligible because their observed t1 score is zero (21 intervention, 15 control). The correct estimand is therefore $\Pr(\text{post}>0\mid\text{observed pre}=0)$, not an all-child post-only probability. The fourth missing t1 nonword score belongs to the control child who also lacks t2.

Within the archive, what little wave missingness exists is **intermittent, not monotone dropout**: the one phonetic-spelling gap is a child observed at t1 and t4 but not t2/t3. This says nothing about the three randomised children absent from the archive, whose missingness pattern is unavailable.

## Recommendation (for the team to decide)

A full ITT claim is not supported by the current data object. Reports should describe the models as **available-case randomised comparisons** or **modified ITT**, show the 57→54 transition by arm, and state that a causal effect for the archived analysis population assumes absence from the dataset is not informative for the treatment effect.

The preferred repair is to recover the three randomised records and include every participant under assigned treatment. If that is impossible, the randomised-phase analyses need a transparent sensitivity over plausible outcomes for the one missing intervention and two missing control children—for example bounded best/worst-case contrasts and a graded pattern-mixture or delta-shift analysis on the probability or item scale. The floor-transition analyses additionally need a sensitivity for unknown baseline-floor eligibility; missing baselines must never be silently classified as zeros. Within the 54-child archive, a separate t4-only sensitivity remains proportionate for the later-wave aligned, level and growth analyses.

## Reproducing this audit

```bash
python scripts/attrition_audit.py
```

The tables above are hand-recorded snapshots and the CSVs are gitignored, so on any change to `data/` or `MEASURES` regenerate them and re-check this note against `analysis_set_audit.csv` and `attrition_audit.csv`.

The command prints both tables and writes `analysis_set_audit.csv` plus `attrition_audit.csv` to `output/audit/` (gitignored). Only the script and this note are committed.
