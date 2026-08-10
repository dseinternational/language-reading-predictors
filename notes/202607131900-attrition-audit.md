<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8). The within-archive counts below were recomputed from `data/rli_data_long.csv` by `scripts/attrition_audit.py`; re-run it to reproduce. Published randomisation counts come from Burgoyne et al. (2012). This note is evidence for the issue #230 §3 decision — it does not itself close the item.

> [!NOTE]
> Substantially revised by a LLM-based AI tool (Codex/GPT-5) following the ITT audit in issue #341.
>
> Participant-flow details from the original CONSORT diagram and missing-data analysis wording added by a LLM-based AI tool (Codex/GPT-5).

# Analysis-set and outcome-attrition audit: is an informative-missingness sensitivity warranted?

Date: 2026-07-13 — relates to issue #230 §3 (measurement and missingness debts).

## The question

Issue #230 §3 flags that the suite's complete-case comparators only handle _covariate_ missingness, not _outcome_ dropout, and asks whether a delta-adjustment or other informative-missingness sensitivity is needed. The first version of this audit considered only the 54 children present in the archived CSV. Issue #341 identified the missing outer denominator: Burgoyne et al. randomised 57 children, 29 to immediate intervention and 28 to waiting control, then analysed 54 after three losses to follow-up. This revision separates those post-randomisation losses from missing cells within the modelling dataset. Trial allocation, follow-up and analysis counts are documented in Burgoyne et al. (2012), DOI [10.1111/j.1469-7610.2012.02557.x](https://doi.org/10.1111/j.1469-7610.2012.02557.x).

## The evidence

### Published randomised population versus archived dataset

| Arm                    | Randomised | Lost to follow-up and absent from archive | Published reason                                                                        | Discontinued intervention but followed and analysed | Analysed and in archive |
| ---------------------- | ---------: | ----------------------------------------: | --------------------------------------------------------------------------------------- | --------------------------------------------------: | ----------------------: |
| Immediate intervention |         29 |                                         1 | Moved school                                                                            |                                                   2 |                      28 |
| Waiting control        |         28 |                                         2 | One moved school; one recorded as “refused to participate in testing, school withdrawn” |                                                   2 |                      26 |
| **Total**              |     **57** |                                     **3** | —                                                                                       |                                               **4** |                  **54** |

The three children lost to follow-up cannot appear in a wave-by-wave audit of `rli_data_long.csv`. Their individual records and outcomes are not encoded in this repository, but the published CONSORT diagram supplies their assigned arms and reasons for loss. The paper separately states that the four children who discontinued the intervention had follow-up measures and were included in its analyses; two came from each arm. The current analyses likewise preserve assigned arm for children represented in the archive rather than selecting on intervention adherence. They remain **available-case modified ITT analyses**, however, because the three lost children are absent from the analysed cohort.

The original paper also states that its ANCOVAs used Mplus full-information maximum likelihood for the small amount of missing data in those analyses. That is relevant precedent for using all available measurements within the paper's analysed cohort, but it does not show that the three children lost to follow-up entered the reported 54-child analysis and does not replace an attrition sensitivity for them.

### Recovered full-randomised screening roster

The checksum-pinned UK Data Service archive now supplies complete screening age, word reading, letter sounds and expressive/receptive vocabulary for all 57 randomised children. The three children excluded from the paper's analysed cohort have no t1 or later measurements in that source, so recovering their screening profiles does not recover their outcomes. The published flow is 57 randomised (29 intervention and 28 waiting control) to 54 analysed (28 and 26). Separately, the archive shows t2 word reading for 53 children (28 and 25): four outcomes are unavailable—three losses to follow-up plus one additional waiting-control child within the 54-child archive. The `lrp-rli-itt-010` release contract therefore retains the 53-outcome t1-baseline model of record and requires an adjacent common-profile screening-baseline bridge, all-57 common-profile MAR standardisation, and a separate factual randomised-arm completion over 29 intervention and 28 control profiles. That completion includes a zero-delta MAR anchor, the documented intervention-non-starter mean-surface no-benefit restriction, a fixed arm-specific delta grid for the one intervention and three control missing outcomes, and the raw sharp bounds. The external archive remains gitignored pending clear redistribution terms. Its local raw CSV retains upstream source identifiers; returned model data and emitted artefacts omit them, and no source-to-repository crosswalk is persisted.

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

Within the archived dataset, outcome-cell missingness is small. That narrower fact must not be used to erase the three children lost to follow-up before this audit begins:

- **The randomised ITT window (t1 → t2) is internally complete** for the seven graded standardised outcomes (W, R, E, L, B, F, T): every archived child with a baseline has a randomised post-score. Word reading has 53 rather than 54 scores at both waves. Phonetic spelling has one child with a t1 score but no t2 score.
- **The DiD crossover chain (t1 → t2 → t3) is complete** for every graded outcome (both the ITT-window and t2→t3 legs are 0). The single t2→t3 gap is one child on the floored nonword-reading measure (N).
- **Only the final maintenance wave (t4) loses anyone**, and at most two children (W, L and F lose 2; the rest lose 0–1). t4 is post-crossover — both arms are treated by then — so it feeds the aligned / level / growth analyses, not the randomised ITT or the DiD contrast.

The nonword t1 count is 50, not 54. Although the regression has no baseline covariate, the floor-transition estimand conditions on an observed t1 zero. Of the 53 children with a t2 score, three have unknown baseline-floor eligibility (one intervention, two control) and are excluded; 36 children are eligible because their observed t1 score is zero (21 intervention, 15 control). The correct estimand is therefore $\Pr(\text{post}>0\mid\text{observed pre}=0)$, not an all-child post-only probability. The fourth missing t1 nonword score belongs to the control child who also lacks t2.

Within the archive, what little wave missingness exists is **intermittent, not monotone dropout**: the one phonetic-spelling gap is a child observed at t1 and t4 but not t2/t3. This says nothing about the unobserved scores of the three randomised children lost to follow-up. Their published reasons are known, but their individual outcome trajectories are unavailable in the repository.

## Recommendation (for the team to decide)

A full ITT claim is not supported by the current data object. Reports should describe the fitted results as **available-case modified ITT estimates**, show `57 randomised → 3 lost to follow-up → 54 analysed` by arm, distinguish those losses from the four children who discontinued the intervention but were followed and retained, and state that a causal effect for the archived analysis population assumes loss to follow-up is not informative for the treatment effect.

Full follow-up-outcome recovery for the three lost children would remain preferable, but the recovered archive contains only their screening assessments. In its absence, randomised-phase analyses need transparent sensitivities over plausible outcomes for the one missing intervention and two missing control children—for example bounded best/worst-case contrasts and a graded pattern-mixture or delta-shift analysis on the probability or item scale. Word reading additionally has one within-archive missing control outcome and now implements that four-profile sensitivity as described above; the other outcomes still require estimand-matched decisions rather than inheriting the word-reading assumptions automatically. The floor-transition analyses additionally need a sensitivity for unknown baseline-floor eligibility; missing baselines must never be silently classified as zeros. Within the 54-child archive, a separate t4-only sensitivity remains proportionate for the later-wave aligned, level and growth analyses.

## Reproducing this audit

```bash
python scripts/attrition_audit.py
```

The tables above are hand-recorded snapshots and the CSVs are gitignored, so on any change to `data/` or `MEASURES` regenerate them and re-check this note against `analysis_set_audit.csv` and `attrition_audit.csv`.

The command prints both tables and writes `analysis_set_audit.csv` plus `attrition_audit.csv` to `output/audit/` (gitignored). Only the script and this note are committed.
