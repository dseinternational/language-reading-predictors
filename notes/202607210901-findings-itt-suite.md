# Intention-to-treat (ITT) suite findings (2026-07-21)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

This note is one of a set of per-family notes from the full 2026-07-21 re-fit of every Bayesian statistical model in the study (production `reporting` configuration, 6 chains × 6000 draws, 89% credible intervals). It covers the **intention-to-treat (ITT) suite** — the 27 single-outcome ITT models. Read the shared index and reading guide first (`notes/202607210900-findings-00-index-and-reading-guide.md`) for the study background, the outcome measures and their maxima, and how to read Bayesian numbers. The related joint models (which fit several outcomes together and estimate the taught-versus-not-taught contrasts) and the block-2 exposure models are reported in a companion note. **All 27 models in this family pass the convergence gate.** Work in progress; all data are preliminary.

## What this family probes

The study is a **waitlist-crossover randomised trial**: children with Down syndrome were randomly assigned either to receive a reading and phonics intervention immediately or to receive it after a wait, and were tested at four timepoints. Because assignment was random, the difference in outcomes between the two assigned arms is a genuine **causal** effect of _being offered the intervention_. That quantity is the **intention-to-treat (ITT) effect**, written τ (tau). It is one of only three causal quantities in the whole statistical suite; everything else across the study is an adjusted association.

The ITT suite answers the headline question directly and one outcome at a time: **did assignment to the intervention change this skill, and by how much?** Each model estimates τ from the first randomised comparison. Crucially, the DAG-faithful (causal-diagram-faithful) core models adjust for **only** the outcome's own baseline score and a linear age term, and they do so purely as _precision_ terms — they sharpen the estimate but are not what identifies the effect. Randomisation is what identifies the ITT effect, so the causal adjustment set is empty; deliberately, no other skill's baseline is conditioned on (conditioning on a downstream skill would bias the randomised comparison). This is the clean single-number answer the other families do not give: the difference-in-differences and gain-factor families re-derive the same effect through the crossover and through period-stacked gain models, and the mechanism and mediation families ask _how_ the effect propagates, but τ here is the primary randomised estimate.

Three sets of robustness members sit alongside the core, each re-estimating the same τ with one extra covariate added to check the headline effect survives — the point is agreement, not novelty:

- **General-ability adjustment (itt-017–024)** adds a block-design cognitive score across the vocabulary family (taught/not-taught receptive and expressive, standardised receptive and expressive) and the two reading anchors (letter-sound knowledge LS, word reading WR).
- **Socio-economic-status (SES) adjustment (itt-013 for WR, itt-113 for LS)** adds parents' post-16 education and the age the child first had books, on the smaller complete-case subset; each is paired with a **matched unadjusted comparator on the identical subset** (itt-014, itt-114) so the effect of the adjustment can be told apart from the effect of restricting the sample.
- **Study-site adjustment (itt-027 for WR, itt-028 for LS)** adds study area.

Every added covariate enters as an _adjusted association_, never as a causal lever; τ remains the only causal coefficient.

Two heavily-floored outcomes — **phonetic spelling (PS, itt-009)** and **nonword reading (NW, itt-011)**, on which most children score zero — take a post-hoc, arm-blind **floor rule**. Their primary estimand is not an items change but a binary **off-floor risk difference**: among children observed at the floor at baseline, the change in the probability of moving above zero. Because the floor threshold was chosen after inspecting the score distribution, these two are flagged **exploratory**.

## How to read these numbers

Full conventions are in the shared reading guide; in brief, and tuned to this family:

- The point estimate is the posterior **median**. Lead with the **interval**, not the point, because with roughly 54 children an estimate that just clears a threshold is on average inflated (the "winner's curse").
- Uncertainty is the **89% equal-tailed credible interval** — "an 89% posterior probability the value lies in this range". 89%, not the customary 95%, is the deliberate house standard.
- Direction is the **tail probability**, P(intervention helps), read directly and never as a p-value. Its **evidence label** is fixed by round odds: _inconclusive_ (below 75%), _suggestive_ (from 75%), _moderate_ (from 91%), _strong_ (from 97%), _very strong_ (from 99%). The label grades the _strength of evidence for a directional claim_; it never describes the _size_ of the effect. (Where the ground-truth sentence attaches a label to a rounded probability, this note quotes that label verbatim, so two effects both shown as "99%" can carry different labels because their unrounded probabilities fall either side of a rung.)
- **Size is a separate claim.** Alongside direction, each model reports the probability the benefit reaches the smallest difference the project judged to matter in practice (δ: 1 item for most skills, 2 items for the larger tests, 10 percentage points for the floored skills), and the probability the effect is "too small to matter either way" (mass inside a region of practical equivalence, ROPE, around zero). These δ thresholds were agreed after the initial results review, so they are read beside the threshold-sensitivity analysis.
- Effects are modelled on a probability (proportion-correct) scale and translated back to **items** by multiplying by the test maximum. For the two floored outcomes the estimand is a **risk difference in percentage points** instead.
- **Causal versus association.** Only τ is causal here. Every covariate — baseline, age, ability, SES, site — is an _adjusted association_ and must not be read as a lever (the Table-2 fallacy). Residual confounding by latent general ability remains for all associations.

## Per-model findings

All numbers below are the model's own gate-interlocked key-findings values. Every one of the 27 models **passes** the convergence gate.

### Core DAG-faithful ITT suite (τ is CAUSAL)

These estimate τ with the empty causal adjustment set (own baseline and linear age as precision terms only). Grouped by outcome, from the clearest signal to the flat measures.

| ID      | Outcome (max items)                      | N   | τ, items — median [89% CI] | P(helps) | Evidence     | Gate |
| ------- | ---------------------------------------- | --- | -------------------------- | -------- | ------------ | ---- |
| itt-007 | LS — letter-sound knowledge (32)         | 54  | +3.5 [+1.7, +5.3]          | 99.9%    | very strong  | pass |
| itt-010 | WR — word reading (79)                   | 53  | +2.4 [+0.7, +4.1]          | 99%      | strong       | pass |
| itt-008 | PA — phoneme blending (10)               | 54  | +1.0 [+0.2, +1.7]          | 98%      | strong       | pass |
| itt-002 | TE — taught expressive vocab (24)        | 54  | +1.5 [+0.4, +2.7]          | 98%      | strong       | pass |
| itt-001 | TR — taught receptive vocab (24)         | 54  | +1.4 [+0.2, +2.5]          | 97%      | moderate     | pass |
| itt-003 | UR — not-taught receptive vocab (12)     | 54  | +0.6 [−0.0, +1.2]          | 94%      | moderate     | pass |
| itt-025 | LF — basic concept knowledge (18)        | 54  | +0.9 [−0.3, +2.0]          | 89%      | suggestive   | pass |
| itt-004 | UE — not-taught expressive vocab (12)    | 54  | +0.3 [−0.3, +1.0]          | 77%      | suggestive   | pass |
| itt-026 | RG — receptive grammar (32)              | 54  | +0.7 [−0.8, +2.1]          | 76%      | suggestive   | pass |
| itt-005 | RV — standardised receptive vocab (170)  | 54  | +0.2 [−3.7, +4.3]          | 54%      | inconclusive | pass |
| itt-006 | EV — standardised expressive vocab (170) | 54  | +0.2 [−3.1, +3.5]          | 53%      | inconclusive | pass |

Reading these outcome by outcome, with the size claim reported separately from direction:

- **Letter-sound knowledge, LS (itt-007)** is the strongest result in the suite: the intervention changed LS by about **+3.5 of 32 letter sounds** (89% credible range +1.7 to +5.3), with a **99.9%** probability the true effect is positive — _very strong_ evidence of benefit. The size claim also lands: the probability the benefit reaches the pre-agreed 2-item threshold is **91%**, with only **9%** of the posterior too small to matter.
- **Word reading, WR (itt-010)**: about **+2.4 of 79 words** (89% +0.7 to +4.1), **99%** probability positive — _strong_ evidence. The probability the benefit reaches the 1-item threshold is **90%** (ROPE mass 9%). A modest but well-supported gain in words actually read.
- **Phoneme blending, PA (itt-008)**: about **+1.0 of 10** (89% +0.2 to +1.7), **98%** probability positive — _strong_. Directionally very clear, but the magnitude is more equivocal: the probability the benefit reaches 1 item is only **49%** (ROPE mass 51%), so a benefit is well supported while its practical size is uncertain — a small (10-item) test amplifies uncertainty when expressed in items.
- **Taught vocabulary, TE and TR (itt-002, itt-001)**: taught expressive TE is **+1.5 of 24** (89% +0.4 to +2.7), **98%** positive, _strong_ (P(benefit ≥ 1 item) 78%); taught receptive TR is **+1.4 of 24** (89% +0.2 to +2.5), **97%** positive, _moderate_ (P(benefit ≥ 1 item) 69%). The words the programme teaches are learned.
- **Not-taught vocabulary, UR and UE (itt-003, itt-004)**, the transfer measures, are weaker. Not-taught receptive UR is **+0.6 of 12** (89% −0.0 to +1.2), **94%** positive, _moderate_ — some generalisation, but the probability of reaching 1 item is only **16%** (ROPE mass 84%), so any transfer is small. Not-taught expressive UE is **+0.3 of 12** (89% −0.3 to +1.0), **77%** positive, _suggestive_, and practically negligible (P(benefit ≥ 1 item) 5%, ROPE mass 95%).
- **Two further language outcomes, LF and RG (itt-025, itt-026)**, fitted with the same empty-adjustment-set core design, extend the suite. Basic concept knowledge LF is **+0.9 of 18** (89% −0.3 to +2.0), **89%** positive, _suggestive_ (P(benefit ≥ 1 item) 43%, ROPE mass 57%); receptive grammar RG is **+0.7 of 32** (89% −0.8 to +2.1), **76%** positive, _suggestive_ (P(benefit ≥ 1 item) 35%, ROPE mass 61%). Both lean positive but are directionally soft and practically small, consistent with the tapering seen across the broader language measures.
- **Standardised vocabulary, RV and EV (itt-005, itt-006)** are flat. Receptive RV is about **+0.2 of 170** (89% −3.7 to +4.3), **54%** positive; expressive EV about **+0.2 of 170** (89% −3.1 to +3.5), **53%** positive. Both are firmly **inconclusive** on direction (with the wide item ranges reflecting the 170-item scale, and ROPE masses of 58% and 67%). This is a genuinely uninformative-about-direction result, not evidence of "no effect": broad standardised vocabulary was not moved over this window.

### Floored code skills — off-floor risk difference (τ is CAUSAL but EXPLORATORY)

For PS and NW the estimand is a binary off-floor risk difference: among children observed at the floor at baseline, the change in the probability of scoring above zero. The floor threshold is post-hoc and arm-blind, so these are exploratory, not prospectively specified.

| ID      | Outcome                | N   | τ, off-floor risk difference — median [89% CI] | P(helps) | Evidence     | Gate |
| ------- | ---------------------- | --- | ---------------------------------------------- | -------- | ------------ | ---- |
| itt-011 | NW — nonword reading   | 36  | +10 pp [−4, +24]                               | 88%      | suggestive   | pass |
| itt-009 | PS — phonetic spelling | 41  | +4 pp [−7, +16]                                | 72%      | inconclusive | pass |

- **Nonword reading, NW (itt-011)**: the intervention changed the chance of coming off the floor by about **+10 percentage points** (89% −4 to +24), **88%** probability positive — _suggestive_. The probability this reaches the pre-agreed 10-percentage-point threshold is **50%** (with a 49% probability the effect is too small to matter either way).
- **Phonetic spelling, PS (itt-009)**: about **+4 percentage points** (89% −7 to +16), **72%** probability positive — _inconclusive_ (P(benefit ≥ 10 pp) 20%, ROPE mass 78%).

Both directions are favourable, but because the floor rule is post-hoc and the subgroups are small (N = 36 and 41), these off-floor headlines should be read as exploratory and not quoted as settled.

### General-ability robustness (itt-017–024; τ is CAUSAL, ability is an association)

Each adds a block-design cognitive score to a core outcome. The block-design coefficient is an _adjusted association_ (not a lever); τ stays the causal quantity. The finding is **agreement**: every adjusted τ lands within a whisker of its core counterpart.

| ID      | Outcome (max)                      | N   | τ, items — median [89% CI] | P(helps) | Evidence     | Corroborates core |
| ------- | ---------------------------------- | --- | -------------------------- | -------- | ------------ | ----------------- |
| itt-023 | LS — letter-sound knowledge (32)   | 54  | +3.5 [+1.7, +5.3]          | 99.9%    | very strong  | ✓ itt-007 (+3.5)  |
| itt-024 | WR — word reading (79)             | 53  | +2.2 [+0.5, +4.0]          | 98%      | strong       | ✓ itt-010 (+2.4)  |
| itt-018 | TE — taught expressive (24)        | 54  | +1.5 [+0.3, +2.6]          | 98%      | strong       | ✓ itt-002 (+1.5)  |
| itt-017 | TR — taught receptive (24)         | 54  | +1.3 [+0.1, +2.4]          | 96%      | moderate     | ✓ itt-001 (+1.4)  |
| itt-019 | UR — not-taught receptive (12)     | 54  | +0.6 [−0.1, +1.2]          | 92%      | moderate     | ✓ itt-003 (+0.6)  |
| itt-020 | UE — not-taught expressive (12)    | 54  | +0.3 [−0.3, +1.0]          | 78%      | suggestive   | ✓ itt-004 (+0.3)  |
| itt-021 | RV — standardised receptive (170)  | 54  | +0.3 [−3.7, +4.4]          | 55%      | inconclusive | ✓ itt-005 (flat)  |
| itt-022 | EV — standardised expressive (170) | 54  | +0.2 [−3.1, +3.5]          | 54%      | inconclusive | ✓ itt-006 (flat)  |

LS stays _very strong_ at +3.5 items, WR _strong_ at +2.2 items, TE _strong_, TR and UR _moderate_, UE _suggestive_, and RV and EV stay flat and _inconclusive_. The headline effects are not an artefact of any cognitive-ability imbalance between the arms.

### SES robustness on the complete-case subset (itt-013/014 for WR, itt-113/114 for LS; τ is CAUSAL, SES is an association)

Each adjusted model is paired with an unadjusted comparator on the **identical** complete-case subset, so the adjustment's effect can be separated from the sample restriction's effect. The samples are roughly halved (N = 33–34).

| ID      | What it is                             | Outcome (max) | N   | τ, items — median [89% CI] | P(helps) | Evidence    |
| ------- | -------------------------------------- | ------------- | --- | -------------------------- | -------- | ----------- |
| itt-014 | WR, unadjusted on the SES subset       | WR (79)       | 33  | +2.4 [+0.5, +4.3]          | 98%      | strong      |
| itt-013 | WR, + SES (parent education, book age) | WR (79)       | 33  | +2.5 [+0.3, +4.7]          | 97%      | moderate    |
| itt-114 | LS, unadjusted on the SES subset       | LS (32)       | 34  | +3.9 [+1.8, +6.0]          | 99.8%    | very strong |
| itt-113 | LS, + SES                              | LS (32)       | 34  | +3.4 [+1.2, +5.6]          | 99%      | very strong |

Word reading is +2.5 items adjusted (itt-013, _moderate_) versus +2.4 unadjusted-on-subset (itt-014, _strong_); letter-sound knowledge is +3.4 adjusted (itt-113, _very strong_) versus +3.9 unadjusted-on-subset (itt-114, _very strong_). The effects survive SES adjustment; the slight softening of itt-013's label to _moderate_ reflects the halved sample, not the adjustment. Both corroborate the core WR (itt-010) and LS (itt-007) results.

### Study-site robustness (itt-027 for WR, itt-028 for LS; τ is CAUSAL, site is an association)

| ID      | What it adds        | Outcome (max) | N   | τ, items — median [89% CI] | P(helps) | Evidence    | Corroborates core |
| ------- | ------------------- | ------------- | --- | -------------------------- | -------- | ----------- | ----------------- |
| itt-027 | + study site (area) | WR (79)       | 53  | +2.6 [+0.8, +4.3]          | 99%      | very strong | ✓ itt-010         |
| itt-028 | + study site (area) | LS (32)       | 54  | +3.5 [+1.7, +5.3]          | 99.8%    | very strong | ✓ itt-007         |

Adjusting for study area leaves both effects essentially unchanged, and if anything sharpens them slightly: word reading +2.6 items (_very strong_) and letter-sound knowledge +3.5 items (_very strong_).

## What to take away

The randomised evidence is internally consistent and points one way. The intervention produces **strong-to-very-strong** benefits on the code-related and directly-taught skills — **letter-sound knowledge (LS, +3.5 of 32, very strong)**, **word reading (WR, +2.4 of 79, strong)**, **phoneme blending (PA, +1.0 of 10, strong)** and **taught expressive vocabulary (TE, +1.5 of 24, strong)** — with **moderate** support for taught receptive vocabulary (TR, +1.4) and not-taught receptive vocabulary (UR, +0.6), tapering to a **suggestive** signal on the softer transfer and language measures (not-taught expressive UE, basic concepts LF, receptive grammar RG). It is **inconclusive and practically negligible on the broad standardised vocabulary tests (RV and EV)** — an uninformative-about-direction result, not proof of no effect. On the floored code skills the direction is favourable but exploratory (nonword reading NW _suggestive_, +10 pp off the floor; phonetic spelling PS _inconclusive_).

These conclusions are robust. Adjusting τ for general ability (itt-017–024), for SES on the matched complete-case subset (itt-013/014, itt-113/114), or for study site (itt-027, itt-028) leaves every headline effect essentially unchanged in direction and rough magnitude — the robustness members corroborate rather than overturn the core. This ITT picture is the anchor the other family notes triangulate against: the difference-in-differences and gain-factor re-analyses recover the same randomised effects through different designs, and the mediation family attributes the word-reading gain to the letter-sound route rather than the vocabulary route — exactly what a code-related benefit that spares broad vocabulary would predict.

Two standing caveats. These are **modified** ITT analyses of observed complete cases (the estimands are "available-case"), so extending the causal reading to children missing from a given model assumes their exclusion is ignorable given the observed pre-treatment information; and with roughly 54 children (and only 33–41 for the SES subsets and the floored subgroups), point estimates are on average magnitude-inflated, so the honest summary is the interval, not the point. Only τ is causal — every covariate (baseline, age, ability, SES, site) is an adjusted association, and residual confounding by latent general ability remains for all of them.
