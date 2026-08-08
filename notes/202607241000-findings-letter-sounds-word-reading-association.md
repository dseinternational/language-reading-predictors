<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Findings — letter-sound knowledge and word reading: levels, counter-cases, and vocabulary in word learning

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8); the Q3 timing correction and #421 closeout were updated by a LLM-based AI tool (Codex/GPT-5). Four associational questions asked of the existing suite and, where the suite had no exactly-matching fit, of a new exploratory probe (`notes/assets/202607241000-ls-wr-association-probe.py`). Numbers follow the house standard — posterior **median** with an **89% equal-tailed** credible interval and the tail probability, never a p-value ([credible-interval standard](202607172359-credible-interval-standard.md)). **Nothing in this note is causal.** Latent general ability (`GA`) is unobserved and unblockable, and every quantity here conditions on contemporaneous, post-treatment skill levels; the only randomisation-licensed effect in the study is the ITT arm. Preliminary — n ≈ 51–53 children, exploratory study.
>
> Supersession warning added by a LLM-based AI tool (Codex/GPT-5).

> [!WARNING]
> **Superseded in part on 2026-08-02.** The model-free descriptions and explicitly clean scratch fits remain a dated exploratory snapshot, but claims below inherited from gate-failed registered mechanism or knee-test fits are withheld under `notes/202608021625-divergence-qualification-policy.md`; nonlinear shape quantities remain zero-divergence-only.

> [!IMPORTANT]
> **#421 closeout (2026-08-08).** All eleven promoted models (`ca-010/011`, `gf-012/013`, `mech-102/103/104/204`, `jm-001/002`, `med-060`) now have reporting-tier fits that pass the automatic convergence gate; the consolidated August findings notes are the current results synthesis. The July power-scaling flags on `jm-001/002` were **not covered by the later targeted prior-refit work**: they remain a warning that parts of the joint fits are weakly likelihood-identified or in prior–data conflict. Under the repository's current release policy, `joint_mechanism` is an observational family and those flags qualify interpretation rather than formally withholding release. This does not make the parameters prior-robust. The Q3 timing defect is also closed below: `RW` and `SP` were rerun as same-wave state covariates.

## Why this note exists

The suite's centre of gravity is the intervention effect. These are the matching **associational** questions, asked in the terms a reader of the reading literature would ask them:

1. How strongly does **letter-sound knowledge (`LS`, YARC-LSK, 32 items)** go with the **level** of **word reading (`WR`, EWRSWR, 79 items)**, once **age**, **hearing** and **non-verbal ability** (WPPSI block design, `blocks`) are held fixed?
2. Does that association survive additionally holding **nonword reading (`NW`, 6 items)** fixed — i.e. is there an `LS`–`WR` link that does not run through measured decoding?
3. What distinguishes the children who **know their letter sounds but read few words**?
4. How does vocabulary relate to word **learning** (gains) — both learning to _read_ words and learning new _words_ — irrespective of letter-sound knowledge?

The suite has close neighbours but no exact match for any of them. `lrp-rli-ca-001` (concurrent) is a levels model but mutually adjusts for five other same-wave skills and excludes floored `NW` as a predictor by design; `lrp-rli-mech-058` / `-101` (mechanism) fit `LS → WR` with hearing, sessions and speech in the adjustment set but carry an **own `WR` baseline** — a change framing, not a level — and do not adjust for non-verbal ability; `lrp-rli-gf-001` (gain factors) has the vocabulary → reading-gain terms but under the full DAG-parent adjustment set rather than a minimal one. So all four were fitted fresh, reusing the registered families' own factories. Where an existing reporting-tier model _does_ answer part of a question — `gf-001`, `mech-056` / `-057` / `-097` / `-098` — it is quoted alongside rather than re-fitted.

## What was fitted

Q1–Q3 are _levels_ questions. Per wave (t1–t4), a between-child Beta-Binomial regression of the `WR` item count on the standardised same-wave logit of `LS`, one row per child, no own baseline and no child random intercept (`factories.build_concurrent_model` — the `lrp-rli-ca-*` machinery). Q4 is a _gains_ question and uses the `gain_factors` factory instead (see that section). Adjusters: standardised age, hearing (`hs`), non-verbal ability (`blocks`, t1-only, broadcast). In the corrected Q3 discriminator fits, phonological memory (`erbto`) and speech production (`deapp_c`) are repeatedly measured states read from the **same wave** as `LS` and `WR`, not t1 values broadcast over later waves. Slopes carry the family's regularising `Normal(0, 0.3)` prior. Sampling is `rep-lite`-equivalent (4 chains × 2000–4000 draws, `target_accept = 0.95`, nutpie).

**Status: exploratory scratch fits, not registered models.** They publish no `config.json` / `diagnostics_summary.json` / report and so bypass the production convergence gate; convergence was checked inline and every fit reported here is clean — **max R-hat 1.000, minimum bulk ESS 1 618, zero divergences** across all 62 fits. They should be promoted to `lrp_rli_ca_0NN` / `lrp_rli_gf_0NN` modules before being cited outside this note.

A scale note that matters for reading the numbers: `LS` enters as a **standardised logit**, so "+1 SD" is a large step near the ceiling — at the cohort mean it is **+11.9 letters at t1, +8.6 at t2, +7.3 at t3, +6.2 at t4** (mean `LS` rises from 14.3 to 23.7 of 32 across the study). Items-scale marginals are average marginal effects on the 79-item `WR` scale.

## Q1 — `LS` and the `WR` level, holding age, hearing and non-verbal ability fixed

| Wave | n   | `LS` slope (logit per +1 SD) | 89% CI         | Items (of 79) | 89% CI        | P(>0)  |
| ---- | --- | ---------------------------- | -------------- | ------------- | ------------- | ------ |
| t1   | 53  | **+0.63**                    | +0.34 to +0.93 | **+4.6**      | +2.1 to +7.6  | ≈1.000 |
| t2   | 53  | **+1.01**                    | +0.78 to +1.23 | **+10.4**     | +7.7 to +12.8 | ≈1.000 |
| t3   | 53  | **+0.71**                    | +0.50 to +0.91 | **+9.1**      | +6.3 to +11.8 | ≈1.000 |
| t4   | 51  | **+0.81**                    | +0.62 to +0.98 | **+11.0**     | +8.6 to +13.1 | ≈1.000 |

**The association is large, present at every wave, and essentially untouched by the adjustment.** The matched _unadjusted_ slopes are +0.63 / +0.99 / +0.76 / +0.83 — within a few hundredths of the adjusted ones at t1 and t2, and _slightly larger_ than them at t3 and t4. Age, hearing and non-verbal ability together explain **none** of the `LS`–`WR` level association in this cohort. Adding the randomised arm as a nuisance term changes nothing (+0.64 / +1.03 / +0.72 / +0.82).

The adjusters' own coefficients say why:

- **Non-verbal ability (`blocks`)** is a weak adjuster of `WR` level once `LS` is in the model: +0.13 (P = 0.78) at t1, +0.06 (0.68) at t2, +0.16 (0.89) at t3, +0.22 (0.97) at t4. Only at t4 does it reach even moderate evidence, and it never rivals `LS`.
- **Hearing** is flat throughout (−0.11 to +0.07; every P between 0.23 and 0.77).
- **Age** matters early and then stops: +0.40 (P = 0.99) at t1 and +0.29 (0.99) at t2, but +0.08 (0.72) at t3 and +0.00 (0.51) at t4. By the end of the study, how old a child is tells you essentially nothing about their word reading once you know their letter sounds.

**The plain-English answer.** Among children of the same age, hearing status and non-verbal ability, those who know more letter sounds read markedly more words at the same point in time — around **+9 to +11 more words** for a step of roughly six to nine letter sounds at the mean, at every wave from t2 onward, with the direction certain to three decimal places. And the "controlling for" does no work: this is not a general-ability effect wearing a letter-sound costume, at least not one that block design can detect. What it emphatically is **not** is evidence that teaching letter sounds _causes_ word reading — the sharpest confound (latent general ability) has no observed handle beyond block design, and block design turns out to be a weak one.

Two shape results already in the suite belong with this: the association **accelerates near mastery** — `mech-058`'s flexible curve puts the knee at ~29.5 of 32 with the slope roughly twice as steep above it, though the knee's credible range (20–29.5) is wide and pressed against the top of the data, and a bounded score on a logit link manufactures some of that bend mechanically ([skill-thresholds note](202607171215-findings-skill-thresholds.md)). And letter sounds and vocabulary do **not** need to be high together — every `L × vocabulary` interaction is mildly _negative_, i.e. additive with a hint of substitution, not synergy.

## Q2 — the same association, additionally holding nonword reading fixed

| Wave | `LS` slope, Q1 | `LS` slope, +`NW` | 89% CI         | Share retained (89% CI) | `NW` slope | P(`NW` > 0) |
| ---- | -------------- | ----------------- | -------------- | ----------------------- | ---------- | ----------- |
| t1   | +0.63          | **+0.61**         | +0.33 to +0.90 | 0.97 (0.47–2.01)        | +0.38      | 0.994       |
| t2   | +1.01          | **+0.75**         | +0.49 to +1.00 | 0.74 (0.47–1.10)        | +0.35      | 0.998       |
| t3   | +0.71          | **+0.57**         | +0.36 to +0.78 | 0.80 (0.47–1.27)        | +0.41      | 0.999       |
| t4   | +0.81          | **+0.53**         | +0.33 to +0.73 | 0.66 (0.40–0.99)        | +0.51      | ≈1.000      |

**Yes — the association survives, and comfortably.** Holding nonword reading fixed attenuates the `LS` slope by roughly a **fifth to a third** at t2–t4 (nothing at t1, where `NW` is almost entirely floored), but what remains is decisively positive at every wave: **P(residual `LS` slope > 0) = 1.000** in all four. So a substantial majority of the `LS`–`WR` level association does **not** pass through measured nonword decoding.

Three qualifications, in order of how much they should change your reading:

1. **`NW` is a 6-item measure with a severe floor** — 80% of children score 0 or 1 at t1, and the median is still only 1 at t4. Conditioning on a badly mismeasured mediator blocks its path only partially, so the residual `LS` slope reported here is an **upper bound** on the true non-decoding channel. The honest statement is therefore "**at most** about two-thirds is non-decoding", not "exactly two-thirds": better measurement of decoding would, if anything, attribute _more_ of the association to it.
2. **The share-retained column is the original sensitivity; the registered joint fit is now the identified readout.** The Q1 and Q2 slopes in this dated table come from _separate_ fits, so their ratio pairs draws under a working-independence assumption. [`lrp-rli-jm-001`](../docs/models/lrp-rli-jm-001/) instead fits word reading and nonword decoding **together at each wave** with an LKJ residual correlation, matched term-for-term to `ca-010` / `ca-011`, and reports `share_retained = β(LS→W | N) / β(LS→W)` as a within-model deterministic built from `β_W − ρ (σ_W/σ_N) β_N`. Its reporting fit passed the convergence gate; the model report, not the separate-fit ratio above, is the authoritative identified result. The two remain useful as a bracket rather than one correcting the other: `jm-001` conditions on the **latent** nonword logit where `ca-011` conditions on the observed count, and the ratio is meaningful only while the unconditional slope stays clear of zero. Power-scaling does not directly diagnose a ratio, so `share_retained` was excluded from that calculation; its component terms nevertheless carried the July weak-likelihood/prior-conflict flags. Treat the identified share as qualified, not as proof that the prior is immaterial.
3. **`NW` is a mediator, not a confounder.** Under the revised DAG the code route runs `LS → NW → WR`, so adjusting for it is deliberately _removing_ a pathway. The Q2 slope is therefore an "`LS`–`WR` association not through measured decoding" — the sight-word / paired-associate channel plus whatever decoding the 6-item `NW` test failed to capture — and, because `NW` is a common effect of `LS` and other reading inputs, conditioning on it can also open a collider path. It is a decomposition to be read alongside Q1, never instead of it.

Meanwhile the `NW` slope itself **rises across the study** (+0.38 → +0.35 → +0.41 → +0.51, every P ≥ 0.994) even after `LS`, age, hearing and ability are held fixed — not strictly monotonically, since t2 dips a little below t1, but the t2→t4 climb is steady. Decoding becomes a progressively better independent marker of word-reading level as the cohort develops.

This sits consistently with the decoding-specificity result, which found `LS → NW` about four times steeper than `LS → WR` on a common logit scale: letter sounds feed decoding hard, decoding feeds word reading, **and** letter sounds retain a large direct association with word reading that decoding does not mediate.

## Q3 — what distinguishes high letter sounds with poor word reading?

### The discrepancy is real, large, and stable

At t4, 30 of 51 children know at least 24 of 32 letter sounds. Their word-reading scores span **4 to 62 words** (quartiles 10 / 22 / 34). So knowing the letter sounds is compatible with almost any word-reading level in this cohort — the discrepancy is not a rounding artefact.

It is also **persistent**. Residualising `WR` on `LS` + age + hearing + ability wave by wave, the residual correlates **r = 0.62–0.83** across every pair of waves, and children in the bottom third of the residual at t2 are still at a median **−7.3 words below expectation** at t4 (middle third −2.0; top third +8.6). This is a stable child characteristic, not measurement noise or regression to the mean.

### What goes with it — partial associations holding `LS` (+ age, hearing, ability) fixed

Each candidate entered one at a time alongside `LS` and the Q1 adjusters; the table gives the candidate's own slope (logit per +1 SD) and P(>0). The three _italicised_ rows are different in kind: age, hearing and non-verbal ability are **always** in the model, so those are the Q1 adjusters' own coefficients repeated here for comparison, not one-at-a-time candidates.

| Candidate                                     | t1             | t2             | t3               | t4               | Reading                                 |
| --------------------------------------------- | -------------- | -------------- | ---------------- | ---------------- | --------------------------------------- |
| Nonword reading (`NW`)                        | +0.38 (0.99)   | +0.35 (1.00)   | +0.41 (1.00)     | **+0.51** (1.00) | strongest and **strengthens** with time |
| Expressive vocabulary (`EV`)                  | +0.46 (0.98)   | +0.34 (0.99)   | **+0.60** (1.00) | +0.50 (1.00)     | strong throughout                       |
| Receptive vocabulary (`RV`)                   | +0.36 (0.97)   | +0.39 (1.00)   | +0.51 (1.00)     | +0.34 (0.99)     | strong throughout                       |
| Phoneme blending (`PA`)                       | +0.47 (1.00)   | +0.17 (0.92)   | +0.27 (0.98)     | +0.34 (1.00)     | present, weakest in the middle          |
| Receptive grammar (`RG`)                      | +0.48 (1.00)   | +0.15 (0.93)   | +0.19 (0.93)     | +0.35 (1.00)     | present                                 |
| Basic concepts (`LF`)                         | +0.38 (0.99)   | +0.35 (1.00)   | +0.27 (0.98)     | +0.23 (0.97)     | present, fading                         |
| Speech production (`SP`, same-wave `deapp_c`) | +0.25 (0.95)   | +0.27 (0.98)   | +0.18 (0.92)     | +0.23 (0.98)     | modest; t3 remains uncertain            |
| Phonological memory (`RW`, same-wave `erbto`) | +0.30 (0.97)   | +0.32 (>0.99)  | +0.29 (0.99)     | +0.26 (0.97)     | clearer and consistent                  |
| _Non-verbal ability (`blocks`)_               | _+0.13 (0.78)_ | _+0.06 (0.68)_ | _+0.16 (0.89)_   | _+0.22 (0.97)_   | **weak — not a discriminator**          |
| _Hearing_                                     | _−0.11 (0.23)_ | _−0.07 (0.27)_ | _+0.07 (0.73)_   | _+0.02 (0.56)_   | **flat — not a discriminator**          |
| _Age_                                         | _+0.40 (0.99)_ | _+0.29 (0.99)_ | _+0.08 (0.72)_   | _+0.00 (0.51)_   | **early only; nothing by t3/t4**        |

Fitting all of them **together** (heavily collinear at n ≈ 51, so read strictly under the Table-2 fallacy and not as a ranking) leaves only two standing at every wave: `LS` itself (+0.46 / +0.73 / +0.34 / +0.36, P ≥ 0.99) and `NW` (+0.24 / +0.24 / +0.32 / +0.39, P 0.94 → 1.000). Expressive vocabulary holds at t3–t4. Correcting the timing asymmetry does **not** make `SP` or `RW` independently resolved in this deliberately over-adjusted joint model: the same-wave `RW` slopes are +0.18 / +0.17 / +0.13 / −0.07 (P = 0.83 / 0.87 / 0.80 / 0.32), and the `SP` slopes are +0.04 / −0.02 / −0.15 / +0.10 (P = 0.59 / 0.45 / 0.16 / 0.78). Their one-at-a-time associations therefore survive the timing correction, especially for phonological memory, but their purported **unique** contributions do not survive conditioning on the correlated skill cluster. Non-verbal ability goes flat or _negative_ in the joint fit (t3: −0.27, P(>0) = 0.033), the classic collinearity/suppression artefact the concurrent family's report warns about, and not a finding.

This targeted rerun comprised 36 fits and was computationally clean: max R-hat 1.000, minimum bulk ESS 5,993 and zero divergences. It changes the interpretation rather than the overall conclusion: the earlier timing asymmetry understated the one-at-a-time `RW` association at t2–t4, but it did **not** explain why `RW` and `SP` collapse in the joint model.

### The subgroup picture

Splitting the 30 high-`LS` children at t4 at their own median word-reading score (22 words), standardised differences (lower-`WR` minus higher-`WR`, Cohen's _d_):

| Larger in the higher-`WR` half             | _d_           | Median lower vs higher |
| ------------------------------------------ | ------------- | ---------------------- |
| Nonword reading                            | **−1.74**     | 1 vs 5 (of 6)          |
| Phonological memory, total (`erbto`)       | **−1.24**     | 22 vs 32 (of 36)       |
| Phonological memory, **nonword** (`erbnw`) | **−1.20**     | 10 vs 15.5             |
| Phonetic spelling (`SPPHON`)               | −0.88         | 43 vs 74 (of 92)       |
| Basic concepts (CELF)                      | −0.88         | 13 vs 15 (of 18)       |
| Receptive vocabulary                       | −0.77         | 51 vs 58               |
| Receptive grammar (TROG)                   | −0.66         | 16 vs 19               |
| Expressive vocabulary                      | −0.59         | 43 vs 45               |
| Phoneme blending                           | −0.52         | 8 vs 9 (of 10)         |
| Speech production (`deapp_c`)              | −0.52         | 222.7 vs 242.5         |
| **Non-verbal ability (block design)**      | **+0.06**     | **16 vs 16**           |
| **Hearing / ear infections**               | −0.16 / +0.38 | —                      |
| **Age (months)**                           | **+0.61**     | **106 vs 96**          |

**The answer, in one sentence: the children who know their letter sounds but read few words are not less able non-verbally, not worse-hearing, and not younger — they are _older_, and they are markedly weaker at holding and reproducing phonological material and at applying the alphabetic code.** The single sharpest discriminator is nonword reading (median 1 of 6 vs 5 of 6); right behind it is **nonword repetition** — the phonological short-term-memory component of the word-repetition task — at _d_ ≈ −1.2. Oral language (vocabulary, grammar, concepts) separates the groups moderately. Block design does not separate them at all (medians identical at 16).

The **age** result deserves emphasis because it inverts the naive reading: the discrepancy group is on average **nine months older**, so this is not "they just haven't got there yet". Combined with `gc-085`'s strongly negative baseline-age effect on word-reading growth rate, older children who have accumulated letter sounds without converting them look like a genuine plateau, not a lag.

### Adversarial checks on Q3

Applying the descriptive-claims discipline this project runs on:

- **Circularity.** Nonword reading and phonetic spelling are themselves written-code outcomes. Saying "poor word readers with good letter sounds are poor at decoding and spelling" is close to redescribing the discrepancy, not explaining it. The genuinely _upstream_ discriminators — the ones that could be causes rather than symptoms — are **phonological memory**, **speech production**, **blending** and **oral language**. That is where the explanatory weight sits, and those effects are moderate, not decisive.
- **Incomplete matching on `LS`.** Within the "high-`LS`" band the lower-`WR` half still knows fewer letter sounds (median 26 vs 30, _d_ = −1.45), so part of the raw subgroup gap is residual `LS`. The whole-cohort residual analysis — which conditions on `LS` properly — agrees with the subgroup table on direction and rank ordering, which is why both are reported.
- **The EWRSWR kink.** Children reading more than 25 words are given additional Test of Single-Word Reading items, so the upper tail of `WR` is a partly different instrument. Of the 30 high-`LS` children at t4, 12 cross that switch — all in the higher-`WR` half. The subgroup contrast is therefore partly a contrast across a measurement-regime boundary. The logit-scale model-based results are less exposed to this than the raw item medians.
- **Floors.** Both written-code discriminators are heavily floored, and increasingly less so over time: `NW` (6 items) is at zero for **72 / 64 / 52 / 40%** of children at t1–t4, and `SPPHON` for **78 / 64 / 57 / 48%**. A "small" association on either is floor-limited, not absent — and the floor is itself moving, so cross-wave comparisons of these two are not on a fixed scale.
- **Sample size.** 15 vs 15 children. Every _d_ in that table has a standard error near 0.4, so treat the ordering as indicative and only the top two or three separations as reasonably resolved.
- **Intervention dose does not appear.** Sessions attended correlates ≈ 0 with the residual at each wave where it is recorded (−0.14 at t1, −0.03 at t2, +0.10 at t3; `attend` is not collected at t4), so the discrepancy is not "these children got less of the programme".

## Q3b — the raw joint distribution at t4, and the counter-cases

The modelled results above are worth checking against the unmodelled numbers, because the cut-points do a lot of work. All figures are t4, on the 51 children with both an observed word-reading and an observed letter-sound score.

### Letter sounds by word-reading band

| Band                  | n   | `WR` range | `WR` median | **`LS` median** | `LS` IQR | `LS` range | ≥20 | ≥24 | ≥28 |
| --------------------- | --- | ---------- | ----------- | --------------- | -------- | ---------- | --- | --- | --- |
| Q1 (lowest quartile)  | 13  | 0–6        | 1           | **16**          | 13–21    | 8–25       | 38% | 8%  | 0%  |
| Q2                    | 13  | 6–10       | 8           | **24**          | 22–25    | 12–27      | 85% | 54% | 0%  |
| Q3                    | 12  | 12–24      | 17.5        | **28.5**        | 26–30    | 18–32      | 92% | 83% | 67% |
| Q4 (highest quartile) | 13  | 26–62      | 36          | **30**          | 29–32    | 17–32      | 92% | 92% | 85% |
| **Bottom 50%**        | 26  | 0–10       | 6           | **21**          | 16–25    | 8–27       | 62% | 31% | 0%  |
| **Top 50%**           | 25  | 12–62      | 26          | **30**          | 28–31    | 17–32      | 92% | 88% | 76% |
| _Whole cohort_        | 51  | 0–62       | 10          | _25_            | 20–30    | 8–32       | 76% | 59% | 37% |

Quartile boundaries are rank-assigned, so the four children tied at 6 words split across Q1/Q2; the median split is clean (nobody scores 11 words). Two things follow.

**The gradient is monotone but heavily compressed at the top.** `LS` medians step 16 → 24 → 28.5 → 30 — the big move (+8) is between the two lowest quartiles, then +4.5, then +1.5 — while `WR` medians go 1 → 8 → 17.5 → 36. Word reading keeps spreading exactly where letter sounds stop moving.

**Twenty-eight letters is very nearly a partition.** **Zero of 26** children in the bottom half of word reading reach 28 letter sounds (their maximum is 27); **19 of 25** in the top half do. Below that the halves overlap heavily — 62% of the bottom half already know ≥20 letters and 31% know ≥24 — so knowing _most_ of the set carries no signal, and knowing essentially _all_ of it carries a great deal.

That is consistent with the accelerating knee (`mech-058`, ~29.5 of 32, slope roughly doubling above it) — and equally consistent with the 32-item YARC-LSK simply running out of room while word reading spans 12 to 62 words in the same band. **These data cannot separate a developmental readiness threshold from the ceiling of the instrument**, which is the deflationary reading the knee result should carry.

### The counter-cases: reading ahead of letter-sound knowledge

At the two medians (`WR` = 10 words, `LS` = 25 of 32):

|               | `WR` < 10 | `WR` ≥ 10 |
| ------------- | --------- | --------- |
| **`LS` < 25** | 19        | **4**     |
| **`LS` ≥ 25** | **6**     | 22        |

41 of 51 concordant, 10 discordant — **6 with more letters than reading, 4 with more reading than letters**.

> [!IMPORTANT]
> **Q3's framing overstates how one-sided the discrepancy is, and the cut-points are why.** Q3 selects on a _fixed_ letter-sound threshold (≥24 of 32 — met by 59% of the cohort) and then splits that group at _its own_ word-reading median (22 words, more than double the cohort median of 10). Both choices push children into the "many letters, poor reading" description. At **matched** median splits the discordance is near-symmetric: **6 one way, 4 the other**. Neither cut-based count is the right instrument — the `LS`-conditional residual is, and it is what Q3's whole-cohort analysis and the ranking below both use.

The four counter-cases, ranked by the continuous residual (observed minus expected `WR` given `LS` + age + hearing + ability):

| Child (id suffix) | `WR` | `LS` | Letters below median | Expected `WR` | Residual  | Rank /51 | `NW`    |
| ----------------- | ---- | ---- | -------------------- | ------------- | --------- | -------- | ------- |
| …62E21F           | 45   | 24   | **1**                | 13.9          | **+31.1** | 1        | 6/6     |
| …A93E23           | 34   | 17   | **8**                | 8.3           | **+25.7** | 2        | **0/6** |
| …EB2CF4           | 15   | 18   | 7                    | 9.2           | +5.8      | 13       | 1/6     |
| …CFE6F2           | 14   | 23   | 2                    | 14.1          | **−0.1**  | 21       | 2/6     |

Only two of the four survive contact with a continuous measure. **…CFE6F2 is not a counter-case at all** — it sits exactly at its model expectation and appears in the cell only because the two medians happen to fall either side of it. **…62E21F** is the cohort's largest over-performer but is one letter below the cut and maxes both nonword reading (6/6) and blending (10/10) — a strong reader on every route, not a letter-sound-poor one. Under a stricter definition (`WR` ≥ median **and** `LS` in the bottom quartile, ≤ 20) only **…A93E23** and **…EB2CF4** qualify: **n = 2**.

### The one clean case: a sight-word reader

**…A93E23** is the genuine article, and the trajectory shows it is not a single-wave blip:

|                  | t1  | t2  | t3  | t4     |
| ---------------- | --- | --- | --- | ------ |
| Word reading     | 17  | 17  | 24  | **34** |
| Letter sounds    | 20  | 22  | 20  | **17** |
| Nonword reading  | 0   | 0   | 1   | **0**  |
| Blending (of 10) | 3   | 4   | 5   | 5      |
| Age (months)     | 120 | 126 | 135 | 141    |

Word reading doubles while letter-sound knowledge drifts _down_ and nonword reading never leaves zero. This is a **sight-word / logographic reader** — a reading vocabulary acquired by whole-word association with no working alphabetic route — and it is the profile the Down syndrome literature describes (Cupples & Iacono 2000, [doi:10.1044/jslhr.4303.595](https://doi.org/10.1044/jslhr.4303.595); Roch & Jarrold 2012, [doi:10.1016/j.jcomdis.2011.11.001](https://doi.org/10.1016/j.jcomdis.2011.11.001)) and exactly what the decoding-specificity work was looking for. This child is also the **oldest in the cohort** (141 months at t4). …EB2CF4 is a milder version: letter sounds flat at 18–19 across all four waves while word reading climbs 0 → 15 and receptive grammar climbs 8 → 27.

### Membership in the counter-case cell is unstable

Thirteen distinct children occupy the cell at least once across the four waves; only two do so at three waves, and none at all four:

```
CFE6F2  X . X X  (3)    FB24DA  . X X .  (2)    6DDCA3  X . . .   5A1654  . . X .
30F04B  X X X .  (3)    C42A8A  X X . .  (2)    EB2CF4  . . . X   5A3EF5  . . X .
                        A93E23  . . X X  (2)    2B8E1F  . X . .   FE472D  . . X .
                                                5D8486  X . . .   62E21F  . . . X
```

This does **not** contradict the earlier finding that the discrepancy is stable (residual r = 0.62–0.83 across waves). The stability is a property of the **continuous residual**; median-split _membership_ churns because most children sit close to the cut. The practical implication is the one stated in the callout above: use the residual, not the cell. By that measure the population of genuine reading-ahead-of-letters children in this cohort is about **two**, and no group profile is reportable at that size — the only defensible output is the case description above.

## Q4 — vocabulary and word _learning_, irrespective of letter-sound knowledge

"Word learning" is ambiguous, so both readings are fitted: **learning to read words** (outcome `WR`) and **learning new words** (the bespoke taught sets `TR` / `TE`). Both are _gains_ questions, so these use the registered `gain_factors` factory — a period-transition ANCOVA (post given that period's own pre) stacked over the three transitions, with a child random intercept, age, non-verbal ability and hearing, and the randomised on-intervention term. Each outcome is fitted **twice**, with skill adjusters `{RV, EV, LS}` and `{RV, EV}`; the difference is exactly "irrespective of letter-sound knowledge". All six fits clean (max R-hat 1.000, min bulk ESS 1 618, zero divergences; 157–161 transition rows over ~53 children).

Only `beta_trt` is causal. Every `gamma` is an adjusted association, and the child random intercept is a partial, shrunken stand-in for between-child heterogeneity — **not** a control for latent general ability.

| Outcome (gains)            | Skill set  | `RV`                           | `EV`                           | `LS`                      |
| -------------------------- | ---------- | ------------------------------ | ------------------------------ | ------------------------- |
| **Word reading** `WR`      | `RV+EV+LS` | +0.16 (−0.07, +0.39) 0.87      | +0.11 (−0.12, +0.35) 0.78      | +0.09 (+0.01, +0.16) 0.97 |
| **Word reading** `WR`      | `RV+EV`    | +0.15 (−0.09, +0.38) 0.84      | +0.17 (−0.06, +0.40) 0.88      | —                         |
| **Taught receptive** `TR`  | `RV+EV+LS` | **+0.38 (+0.18, +0.58) 0.999** | +0.03 (−0.18, +0.24) 0.59      | +0.05 (−0.00, +0.11) 0.94 |
| **Taught receptive** `TR`  | `RV+EV`    | **+0.40 (+0.21, +0.59) 1.000** | +0.09 (−0.12, +0.29) 0.75      | —                         |
| **Taught expressive** `TE` | `RV+EV+LS` | **+0.28 (+0.07, +0.49) 0.983** | **+0.31 (+0.07, +0.56) 0.981** | +0.07 (+0.01, +0.14) 0.97 |
| **Taught expressive** `TE` | `RV+EV`    | **+0.30 (+0.10, +0.51) 0.989** | **+0.37 (+0.13, +0.61) 0.992** | —                         |

(Median, 89% CI, P(>0); logit per +1 SD of the period-baseline logit.)

### The headline is a dissociation

**Vocabulary strongly predicts learning new spoken words, and barely predicts learning to read them.** Broad receptive vocabulary is the dominant predictor of taught-receptive-word gains (**+0.38**, P = 0.999, very strong); both vocabulary measures predict taught-expressive gains (+0.28 and +0.31, both P ≈ 0.98). Against word-reading gains the same measures reach only **+0.16 / +0.11**, with intervals straddling zero and P = 0.78–0.87 — _suggestive_ at best on the project's evidence ladder.

**Holding letter sounds fixed makes almost no difference.** Dropping `LS` moves the vocabulary slopes by 0.01–0.06 in every one of the three outcomes (`WR`: +0.16/+0.11 → +0.15/+0.17; `TR`: +0.38/+0.03 → +0.40/+0.09; `TE`: +0.28/+0.31 → +0.30/+0.37). So the answer to "irrespective of letter-sound knowledge" is: **the vocabulary associations are what they are with or without it** — letter-sound conditioning is not what makes vocabulary look weak for reading gains, and not what makes it look strong for word learning. That is what the DAG predicts, since `LS` is neither a parent of `RV`/`EV` nor a confounder of the vocabulary → reading path. (Conditioning on `LS` _would_ open `EV ← SP → LS ← IG → WR`, but `IG` is in the model as the treatment term, which closes it.)

### Levels versus gains: the sharpest contrast in this note

The same vocabulary measures behave completely differently depending on whether the question is about a level or a change:

| Vocabulary → word reading                | Estimate (logit per +1 SD)               | Evidence                    |
| ---------------------------------------- | ---------------------------------------- | --------------------------- |
| **Level**, given `LS` (Q3)               | `RV` +0.33 to +0.51; `EV` +0.34 to +0.60 | P 0.97–1.000, very strong   |
| **Gain**, given `LS` (Q4)                | `RV` +0.16; `EV` +0.11                   | P 0.87 / 0.78, suggestive   |
| **Gain**, full DAG-parent set (`gf-001`) | `RV` +0.05; `EV` +0.02                   | P 0.64 / 0.56, inconclusive |

Vocabulary **tracks** where a child's word reading has got to, but adds little to **how fast it moves** once the child's own reading baseline is in the model — and once the taught-vocabulary and code measures are added too (`gf-001`), it goes to nothing. This is the cross-sectional-versus-longitudinal gap that the levels-only reading of Q1–Q3 would hide, and it matters for interpretation: a strong concurrent correlation with vocabulary is not evidence that vocabulary is driving reading progress. It is consistent with the mechanism family's independent verdict — `mech-056` (`RV → WR`) +2.9 items over the full exposure range (89% −2.6 to +8.2, P = 0.80) and `mech-057` (`EV → WR`) +5.3 items (−0.5 to +10.9, P = 0.93), neither resolved — and with the GP knee tests, where the two standardised vocabulary curves are essentially flat (`increasing_frac` ≈ 0.57–0.59; [skill-thresholds note](202607171215-findings-skill-thresholds.md)).

### Two more things the same fits say

- **Age is a consistent brake on reading gains** (`gamma_A` = −0.14 with `LS`, −0.16 without; P(negative) ≈ 0.997–0.999), and a weaker one on taught-word gains (−0.06 to −0.12). This corroborates `gc-085`'s negative baseline-age effect on word-reading growth rate and the Q3 finding that the discrepancy group is _older_.
- **Non-verbal ability splits the two kinds of learning.** Block design is flat for word-reading gains (+0.01, P = 0.60) but mildly positive for taught-word gains (+0.08 to +0.10, P = 0.92–0.95) — the opposite of the folk expectation that non-verbal ability gates reading specifically.

### Caveats

Adjusted associations throughout, with latent general ability unblocked. The taught-word measures are short bespoke sets scored against what was actually taught, so `TR`/`TE` gains partly reflect programme exposure and item overlap with a child's existing vocabulary — a child who already knows more words may find the taught set easier for reasons that are not "learning ability". `gf-001`'s tighter nulls and this probe's looser suggestive slopes differ only in the adjustment set, which is a Table-2-fallacy hazard: they answer different conditional questions and neither is "the" vocabulary effect. And 157–161 rows over ~53 children is the same small sample throughout — the taught-vocabulary results are the only ones here strong enough to survive that on their own.

## What we can and cannot say

**Can say.** Letter-sound knowledge is by a wide margin the strongest same-wave correlate of word-reading level in this cohort; age, hearing and non-verbal ability account for none of it; a large majority of it does not run through measured nonword decoding; the children who break the pattern — good letter sounds, poor word reading — are distinguished by phonological memory, decoding and, secondarily, oral language, not by non-verbal ability, hearing or youth; and vocabulary predicts learning new _spoken_ words strongly but learning to _read_ words only weakly — the same either way once letter sounds are held fixed.

**Cannot say.** That any of this is causal. `GA` is latent and block design is a poor proxy for it; every coefficient conditions on post-treatment skill levels; and the discrepancy-group comparison is a 15-vs-15 descriptive split at a single wave. Nothing here licenses "teach more letter sounds and word reading follows", nothing licenses "train phonological memory and the discrepancy closes", and the Q4 vocabulary results are associations with word _learning_, not evidence that vocabulary drives it.

## Models selected for formal build and fit

This section preserves the selection rationale that produced #421; it is no longer a build backlog. The eleven resulting models are implemented and have gate-passing reporting fits. **`lrp-rli-mech-090` (`RW → WR`) and `lrp-rli-mech-190` (`PA → WR`) already existed** — an earlier draft of this list proposed them, wrongly. `mech-090` put phonological memory at **+0.10 logit per +1 SD (89% +0.00 to +0.20, P = 0.952)**, which is why #421 added its decoding counterpart.

### Tier 1 — implemented with existing factories

| Built model        | Family         | What it estimates                                     | Closeout                                                                                                                                                                                                                   |
| ------------------ | -------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lrp-rli-ca-010`   | `concurrent`   | `WR` level ~ `LS` + age + hearing + ability, per wave | Reporting fit passes; the consolidated concurrent note reports an adjusted `LS` slope of +0.77.                                                                                                                            |
| `lrp-rli-ca-011`   | `concurrent`   | as `ca-010`, plus `NW`                                | Reporting fit passes; the consolidated note reports `LS` +0.59 and `NW` +0.39.                                                                                                                                             |
| `lrp-rli-gf-012`   | `gain_factors` | `TR` gains with `skill_symbols = ("R", "E")`          | Reporting fit passes; the randomised treatment marginal is +1.3 items (+0.1 to +2.4). The vocabulary coefficients remain downstream descriptive associations.                                                              |
| `lrp-rli-gf-013`   | `gain_factors` | `TE` gains with `("TR", "R", "E")`                    | Reporting fit passes; the randomised treatment marginal is +1.3 items (+0.2 to +2.4). The vocabulary coefficients remain downstream descriptive associations.                                                              |
| `lrp-rli-mech-102` | `mechanism`    | **`RW → NW`** (phonological memory → decoding)        | Reporting fit passes and reports +2.5 nonword items (+1.7 to +3.3) across the exposure range. The joint Q3 coefficient remains unresolved, so this is a separate adjusted association, not evidence of a unique mechanism. |
| `lrp-rli-mech-103` | `mechanism`    | **`SP → NW`** (speech production → decoding)          | Reporting fit passes and reports +2.3 nonword items (+1.4 to +3.1). The joint Q3 coefficient remains unresolved, so the same qualification applies.                                                                        |

Note `gf-1NN` is already the treated-only-companion convention (`gf-101` is `gf-001`'s companion), which is why the taught-vocabulary additions take `gf-012`/`gf-013` rather than `gf-109`/`gf-110`.

### Tier 2 — implemented, with moderation unresolved

| Built model                | Family      | What it estimates                                                  | Closeout                                                                                                                                                                                                                                                                                                            |
| -------------------------- | ----------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lrp-rli-mech-104` / `204` | `mechanism` | `LS → WR` **moderated by `RW`** / matched no-interaction companion | Both reporting fits pass. `gamma_int` is −0.01 (−0.11 to +0.10, P>0 = 0.46), so there is no resolved phonological-memory moderation. Exact-refit repair makes the nested LOO comparison valid, but it remains inconclusive (`elpd_diff` −1.0); that is a reliability result, not evidence for either specification. |
| `lrp-rli-mech-063` / `163` | `mechanism` | `LS → WR` **moderated by `NW`** / matched no-interaction companion | This descendant-moderator question was implemented under #404 rather than duplicated as the proposed `mech-105`. It remains descriptive and floor-limited; the August mechanism synthesis finds no resolved moderation.                                                                                             |

### Tier 3 — factory and pipeline work (implemented)

1. ~~**A joint `{WR, NW}` levels model.**~~ **Built** (#421 Tier 3 (1)), as two models rather than one, because the two quantities live on different parameterisations. [`lrp-rli-jm-001`](../docs/models/lrp-rli-jm-001/) is the per-wave levels model, matched to `ca-010` / `ca-011`, and reports Q2's share-retained as an identified within-model deterministic. [`lrp-rli-jm-002`](../docs/models/lrp-rli-jm-002/) is the phase-stacked ANCOVA companion, matched to `mech-096` / `mech-101`, and reports the Tier-1 Δ on the parameterisation the [decoding-specificity note](202607172358-findings-decoding-specificity.md) defines it on. Both reporting fits pass. Their Δs differ because they answer different questions: the levels Δ is near zero or negative, whereas the conditional-change Δ is +0.81 [0.50, 1.14]. Both carry an LKJ cross-outcome dependence block; the remaining power-scaling flags are interpretive qualifications, not unresolved convergence failures.
2. ~~**A floor-tolerant second mediator leg.**~~ **Built as [`lrp-rli-med-060`](../docs/models/lrp-rli-med-060/)** (the proposed `med-081` alias was already occupied by `lcsm-081`). Its second mediator is a Bernoulli off-floor nonword leg, so the chained `LS → NW → WR` decomposition is fittable without fabricating an autoregressive nonword baseline. The reporting fit passes, but the nonword-leg indirect effect is essentially zero and remains floor-limited; the module also retains an explicit methodological-sign-off flag on the adjustment set and off-floor NIE definition.

### Explicitly not worth building

- **A mixture / latent-class model for the sight-word subgroup.** By the continuous residual there are about **two** such children; mixture models need orders of magnitude more.
- **An item-level LSK measurement model** to separate a real readiness threshold from the 32-item ceiling (the Q3b ambiguity). **Blocked by data, not modelling** — there is no item-level data, as [design-lessons](202607172345-design-lessons-for-future-studies.md) already records.
- **More GP knee models.** The [skill-thresholds note](202607171215-findings-skill-thresholds.md) already swept six exposures and found only `LS` has a resolved knee, and Q3b shows even that one is confounded with the instrument ceiling. Another knee model cannot break that tie.

### Probe timing closeout (completed 2026-08-08)

`RW` and `SP` now enter the Q3 partials at the **same wave** as the skill measures, through per-row state covariates rather than t1 values broadcast across all waves. The one-at-a-time `RW` association becomes clearer (+0.30 / +0.32 / +0.29 / +0.26, P = 0.97 / >0.99 / 0.99 / 0.97); `SP` remains modest (+0.25 / +0.27 / +0.18 / +0.23). Neither becomes independently resolved in the heavily collinear joint fit. Thus the timing asymmetry was real and worth correcting, but it was **not** the reason the two terms collapsed after mutual adjustment.

**And one non-analytic follow-up:** re-examine the two counter-case children (…A93E23, …EB2CF4) with the study team. At n = 2 this is a case series, not a subgroup — but it is the study's only direct evidence on the sight-word route, and the team may know things about those children the data do not carry.

## Cross-references

- Probe script (all fits and CSVs): [`notes/assets/202607241000-ls-wr-association-probe.py`](assets/202607241000-ls-wr-association-probe.py).
- Concurrent (levels) family, mutually-adjusted view: [`notes/202607210911-findings-concurrent.md`](202607210911-findings-concurrent.md); models `docs/models/lrp-rli-ca-001/`, `lrp-rli-ca-002/`.
- Letter-sound curve shape and the knee: [`notes/202607171215-findings-skill-thresholds.md`](202607171215-findings-skill-thresholds.md); models `docs/models/lrp-rli-mech-058/`, `lrp-rli-mech-101/`.
- Is letter-sound knowledge used for decoding: [`notes/202607172358-findings-decoding-specificity.md`](202607172358-findings-decoding-specificity.md).
- Causal direction workstream: [`notes/202607172100-reverse-mediation-wr-ls-direction-spec.md`](202607172100-reverse-mediation-wr-ls-direction-spec.md), [`notes/202607172230-riclpm-direction-plan.md`](202607172230-riclpm-direction-plan.md).
- Base DAG: [`dag/dag-language-reading.dagitty`](../dag/dag-language-reading.dagitty).
- Measure symbols: [`notes/202607211200-measure-abbreviations-standard.md`](202607211200-measure-abbreviations-standard.md).
