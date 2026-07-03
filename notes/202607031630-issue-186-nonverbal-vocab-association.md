# Baseline non-verbal ability → vocabulary: an adjusted association (issue #186, Q4)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

**Question (Q4, collaborator request — issue #186):** does baseline non-verbal ability (block design) predict receptive and expressive vocabulary?

**Short answer:** after adjusting for the child's own baseline vocabulary, age, and randomised arm, baseline non-verbal ability has a **modest positive _incremental_ association with the intervention's taught vocabulary** (suggestive-to-moderate) and **essentially none with the standardised vocabulary tests**. This is an adjusted association, not a causal effect (see framing below).

## Vehicle — no new model needed

Q4 is already estimated by the ability-adjusted ITT family **LRPITT17–22** (#119), which adds block design (`blocks`, the WPPSI-III Block Design subtest — recorded at t1, complete for all 54 children) as a linear covariate `gamma_blocks · z(blocks)` on top of the child's own baseline and linear age, across the six vocabulary outcomes: taught receptive/expressive (TR/TE = `b1retau`/`b1extau`), not-taught receptive/expressive (UR/UE = `b1rent`/`b1exnt`), and standardised receptive/expressive (R/E = ROWPVT/EOWPVT). `gamma_blocks` — the coefficient per +1 SD of block design — is exactly the Q4 estimand; those models simply fit it as a precision/robustness term rather than surfacing it. `scripts/ability_vocab_association.py` consolidates it across the six models.

## Results (reporting tier)

Coefficient on +1 SD of block design (logit scale), adjusted for own baseline, age, and arm. Direction is read from `P(β>0)` per the evidence-language policy (#179), never from whether an interval excludes zero — every 90% interval below includes zero.

| Outcome                               | `gamma_blocks` (logit) | 90% ETI          | P(β>0) | evidence     | items / +1 SD |
| ------------------------------------- | ---------------------: | ---------------- | -----: | ------------ | ------------: |
| TE — taught expressive (`b1extau`)    |                 +0.139 | [−0.014, +0.291] |  0.934 | moderate     |         +0.68 |
| TR — taught receptive (`b1retau`)     |                 +0.102 | [−0.037, +0.242] |  0.887 | suggestive   |         +0.55 |
| UR — not-taught receptive (`b1rent`)  |                 +0.113 | [−0.053, +0.281] |  0.866 | suggestive   |         +0.27 |
| E — standardised expressive (EOWPVT)  |                 +0.016 | [−0.077, +0.108] |  0.612 | inconclusive |         +0.41 |
| R — standardised receptive (ROWPVT)   |                 +0.003 | [−0.102, +0.107] |  0.519 | inconclusive |         +0.09 |
| UE — not-taught expressive (`b1exnt`) |                 −0.021 | [−0.210, +0.170] |  0.426 | inconclusive |         −0.06 |

The positive signal concentrates on the **taught** block-1 vocabulary (TE moderate, TR suggestive) and not-taught receptive (UR suggestive); the **standardised** tests (R/E) and not-taught expressive (UE) are inconclusive.

## How to read this — an adjusted association, not a cause

Per the locked DAG (`notes/202606231600-dag-revision-consolidated.md`), block design is an **off-DAG, pre-randomisation child covariate**, and latent general ability (`GA`) is the unobserved common cause of block design, vocabulary, and the other skills. So `gamma_blocks` is an **adjusted association — never a causal "non-verbal ability drives vocabulary"**: block design is essentially an ability proxy and the association is confounded by `GA` (not point-identified). Two further points to hold:

- **Incremental, not marginal.** Because the model already conditions on the child's own baseline vocabulary, `gamma_blocks` is the incremental predictive value of non-verbal ability **beyond baseline vocabulary and age** — not the raw correlation. The raw (marginal) block-design ↔ vocabulary correlation is larger (ability is broadly prognostic, most strongly for vocabulary — the reason it makes a useful precision adjuster), but it is largely _shared with_ baseline vocabulary; once baseline vocabulary is in the model, little independent non-verbal signal remains for the standardised tests.
- **Why taught > standardised is plausible.** The taught block-1 words are the intervention's own targets; a child's general ability may shape uptake of newly taught items more than their standing on a broad standardised test whose variance baseline vocabulary already captures. This is a hypothesis, not an identified mechanism.

## Caveats

- **n = 54**, single cohort; every 90% interval includes zero — these are suggestive / moderate _directional_ statements, not decisive effects.
- **Adjusted association**, `GA`-confounded, not causal (above).
- Reporting-tier fits (6 chains × 6000); `gamma_blocks` is stable dev → reporting.

## Follow-ups (planned)

- **GB corroboration** (Phase 2): add block design as a forward-filled baseline covariate to the vocabulary gradient-boosting models (LRPGBG/LRPGBL 01–06) and read its permutation/SHAP importance as a nonparametric cross-check of `gamma_blocks`.
- **Byrne cohort** (Phase 3): the analogue `basmat` → `bpvs` (receptive), gated on the #164 Byrne data decisions and needing a covariate-adjusted vehicle on the historical panel (`basmat` is wave-3+; Byrne has no expressive-vocabulary measure).
