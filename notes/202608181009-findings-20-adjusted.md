> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings: the `adjusted` family — which baseline measures track later progress?

**Read `findings-00-overview` first.** This note covers the 7 models in the `adjusted` family. **Nothing here is causal.**

## The data

**Both studies.** One model uses the RLI trial (51 children); six use the historical Reading, Language and Memory cohort (22–84 children).

Most use **one row per child**: a set of baseline measures, and the gain on some outcome between the first and a later wave — so the time dimension is **collapsed into a single change score**. One model (`rlm-adj-006`) instead pools annual transitions across waves 1 to 5, giving 84 children and 225 rows.

## What the model is for

The question is: **given several baseline measures at once, which ones track later progress?**

The distinguishing feature is _mutual adjustment_. Each predictor's coefficient is estimated holding the others constant, so it describes what that measure adds beyond the rest, rather than its raw relationship with the outcome. The family also reports the simple unadjusted association alongside, because the two often differ and the difference is informative.

This is the same question the `horseshoe` family asks with a heavily sceptical prior. Here the coefficients are estimated more conventionally, so this family is more willing to report a value — and correspondingly more exposed to overfitting.

## What was found

The results are dominated by one pattern:

| Model         | Cohort                         | Outcome                   | Strongest adjusted predictor | Association per +1 SD       |
| ------------- | ------------------------------ | ------------------------- | ---------------------------- | --------------------------- |
| `adj-065`     | RLI trial                      | word-reading gain         | **Age at t1**                | **−2.9 items** [−4.3, −1.3] |
| `rlm-adj-006` | historical                     | word-reading progress     | **Age**                      | **−2.7 items**              |
| `rlm-adj-005` | historical                     | verbal-memory gain        | **Age**                      | **−1.1 items**              |
| `rlm-adj-003` | historical                     | receptive-vocabulary gain | Verbal reasoning             | +0.8 items                  |
| `rlm-adj-004` | historical                     | receptive-grammar gain    | Verbal reasoning             | +0.4 items                  |
| `rlm-adj-002` | historical, Down syndrome only | word-reading gain         | Receptive vocabulary         | −1.9 items                  |

**Age is the strongest adjusted predictor of gain in both cohorts, and it is negative in both.** Older children gained less. This replicates across two entirely separate studies, decades apart, with different measures — which makes it unlikely to be a quirk of one dataset.

## What the age result probably means

The interpretation matters, because "older children gain less" invites an unwarranted developmental story.

The most likely explanations are structural rather than substantive. Older children start higher, and on a bounded test starting higher leaves less room to gain — a **ceiling effect**. Older children measured at baseline are also subject to **regression to the mean** in the same way as any baseline predictor of change. And developmental gain on these instruments is generally faster earlier, so a raw-score increment naturally shrinks with age.

Together these are sufficient to produce the observed pattern without any claim that being older impedes learning. It is also the sort of relationship that changes sign under a different outcome scale.

The verbal-reasoning results in the historical cohort (+0.8 and +0.4 items) point the expected way — more able children gaining somewhat more — but are small.

**`rlm-adj-002` deserves a caution of its own.** It fits **22 children** (the Down syndrome subgroup) with a mutually adjusted predictor set. Twenty-two children cannot support a multi-predictor model in any meaningful sense: the coefficients will be unstable and highly sensitive to individual children. Its headline (receptive vocabulary, −1.9 items) should not be quoted as a finding. The `horseshoe` family's flat rankings are the more honest reading of the same question.

## Set this beside the horseshoe result

The `horseshoe` family asked essentially this question with a prior that aggressively shrinks weak predictors, and found **nothing survived** for gain outcomes in either cohort.

That is the fair overall conclusion. This family reports larger coefficients because it does not shrink them, not because it found more signal. Where the two disagree, the horseshoe result is the more trustworthy guide to whether a predictor is real.

## What these models cannot tell you

**No predictor here is a lever.** Age especially: nothing can be done about it, and the coefficient is not a claim about what ageing does.

**Mutually adjusted coefficients are not independent effects.** Given the very high correlations between skill domains documented in the `corr_factor` note, the predictors overlap heavily and their individual coefficients are correspondingly unstable.

**Collapsing to a single gain discards the trajectory shape** and inherits regression to the mean.

**Small samples.** Several models fit fewer than 70 children and one fits 22.

## The withheld model

`rlm-adj-001` (historical word-reading gain) is **withheld at the inputs stage** because one of its measures has no confirmed maximum score. Its siblings `rlm-adj-003`, `004` and `005` cleared the gate — their titles note "confirmed-input" — because they use only measures whose denominators are established.

## Model inventory

Six of seven pass and are publishable: `adj-065` (RLI trial), `rlm-adj-002` (Down syndrome only, 22 children), `rlm-adj-003` (receptive vocabulary), `004` (receptive grammar), `005` (verbal memory), `006` (pooled annual transitions). `rlm-adj-001` is withheld.
