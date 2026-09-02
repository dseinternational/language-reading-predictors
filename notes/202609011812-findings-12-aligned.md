> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

# Findings: the `aligned` family — a per-protocol view aligned by when the intervention started

**Read `findings-00-overview` first.** This note covers the 10 models in the `aligned` family. **No quantity in this family is causal**, including the group contrast — this is the only family that reports an immediate-versus-waiting-list contrast and declines to call it causal. All 10 pass the convergence gate with zero divergences and all are publishable (2026-09-01 rebuild); the phoneme-blending link pair `al-006`/`al-306` is complete.

## The data

**RLI trial only**, arranged differently from anywhere else. Each child is **aligned to their own intervention onset**: the immediate group is measured from timepoint 1 to 3, the waiting group from timepoint 2 to 4, both spanning about 40 weeks — the full programme. Each child contributes one row (52–54 children), with the onset score as the baseline; there is no random intercept.

## Why the comparison is not randomised, despite looking like one

Randomisation makes two groups comparable at a common moment. Aligning by onset destroys that: the waiting-list children are older when their window starts and their window sits later in calendar time. The cohort contrast therefore confounds the intervention with age-at-onset and timing; adjustment for age at onset and ability cannot restore a randomised comparison. The dose variant adds cumulative sessions, a collider, and is a sensitivity view only.

## What was found

| Outcome                               | Cohort contrast (immediate − waiting) | 89% range         | P(favoured direction) | Evidence     |
| ------------------------------------- | ------------------------------------- | ----------------- | --------------------- | ------------ |
| Receptive vocabulary (R)              | +2.6 items                            | −1.8 to +7.1      | 0.83 (positive)       | suggestive   |
| Letter-sound knowledge (L)            | **+2.2 items**                        | +0.2 to +4.2      | 0.96 (positive)       | moderate     |
| Word reading (W)                      | +2.1 items                            | −0.5 to +4.7      | 0.91 (positive)       | suggestive   |
| Word reading, dose variant (`al-101`) | +2.2 items                            | −0.4 to +4.8      | 0.91 (positive)       | moderate     |
| Phoneme blending (B), ordinary link   | +0.3 items                            | −0.6 to +1.2      | 0.70 (positive)       | inconclusive |
| Phoneme blending (B), guessing floor  | +0.2 items                            | −0.5 to +0.9      | 0.66 (positive)       | inconclusive |
| Basic concept knowledge (F)           | −0.6 items                            | −1.7 to +0.4      | 0.82 (negative)       | suggestive   |
| Phonetic spelling (P), off-floor      | −1.1 pp                               | −13.1 to +10.9 pp | 0.56 (negative)       | inconclusive |
| Receptive grammar (T)                 | −1.4 items                            | −3.1 to +0.3      | 0.91 (negative)       | moderate     |
| Expressive vocabulary (E)             | −3.1 items                            | −6.9 to +0.7      | 0.90 (negative)       | suggestive   |

Two rows with the same posterior probability carry different labels (word reading 0.91 "suggestive", its dose variant 0.91 "moderate", receptive grammar 0.91 "moderate"): the labels are assigned from the unrounded probabilities, which straddle the 0.91 rung.

Only letter-sound knowledge clears zero. Word reading sits close to the randomised estimates (+2.1 against +2.4 in `itt`) with an interval that spans zero; letter sounds is smaller than the randomised +3.5. Receptive vocabulary comes out at +2.6 and expressive vocabulary at −3.1 from largely the same children over aligned windows, and receptive grammar leans negative at moderate evidence: the immediate cohort, which is younger at onset, ended its 40-week window lower on grammar and expressive vocabulary than the older waiting cohort ended theirs. That is the age-at-onset and timing confound the design cannot remove, and it is a reason not to read any of these cohort contrasts as effects.

The covariates behave as in the other families: the onset score is the dominant term everywhere, and in the dose variant age at onset is negative (−0.17 logits per SD, P(negative) = 0.97) while cumulative sessions are inconclusive (+0.04, 89% −0.12 to +0.19).

## How to read this family

A **descriptive sensitivity view**. Where it agrees with the randomised families (letter sounds and word reading positive) that is compatible with them but not independent corroboration; where it disagrees (grammar, expressive vocabulary) the disagreement is compatible with the confounding built into the design. For the causal question, prefer the randomised-window estimates.

## Model inventory

All 10 pass the convergence gate with zero divergences and are publishable: `al-001` (W), `002` (R), `003` (E), `004` (L), `005` (P, off-floor with the binary off-floor-at-onset baseline), `006` (B), `007` (F), `008` (T), `101` (W, cumulative-session dose variant) and `306` (B, guessing-floor link companion, released as a pair with `006`).
