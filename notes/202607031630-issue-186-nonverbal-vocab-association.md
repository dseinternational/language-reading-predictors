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

## GB corroboration (nonparametric)

A gradient-boosting cross-check (`scripts/blocks_vocab_gb_diagnostic.py`) refits the six vocabulary **level** models with block design added to the predictor set (it is normally excluded as t1-only; the loader now broadcasts it per child) and reports where block design ranks by out-of-fold permutation importance (dev tier):

| Model    | Target                           | blocks rank (of 33) | perm. importance | marginal ρ |
| -------- | -------------------------------- | ------------------: | ---------------: | ---------: |
| lrpgbl06 | eowpvt (standardised expressive) |                   3 |            0.686 |      +0.56 |
| lrpgbl04 | b1exnt (not-taught expressive)   |                   3 |            0.088 |      +0.55 |
| lrpgbl03 | b1rent (not-taught receptive)    |                   7 |            0.017 |      +0.45 |
| lrpgbl01 | b1retau (taught receptive)       |                  10 |            0.020 |      +0.50 |
| lrpgbl05 | rowpvt (standardised receptive)  |                  10 |            0.093 |      +0.52 |
| lrpgbl02 | b1extau (taught expressive)      |                  32 |           −0.013 |      +0.47 |

- **Block design carries marginal predictive signal.** It ranks mid-pack or better in five of six models (3–10 of 33) and correlates positively with every vocabulary target (marginal ρ ≈ 0.45–0.56) — consistent with non-verbal ability being broadly prognostic. It is not an unused predictor.
- **Marginal (GB) vs incremental (Bayesian), as expected.** The GB permutation importance does not privilege the child's own baseline vocabulary as an anchor, so it reflects the _marginal_ contribution (larger, shared with baseline); the near-null Bayesian `gamma_blocks` shows that, incremental to baseline, little independent signal remains. The dev-tier ranks are also **noisy** (permutation SD ≥ mean in places, cv = 5) and do not cleanly reproduce the Bayesian taught-vs-standardised ordering — a reporting-tier GB fit (cv = 51 + SHAP direction) would sharpen the per-outcome comparison.

## Caveats

- **n = 54**, single cohort; every 90% interval includes zero — these are suggestive / moderate _directional_ statements, not decisive effects.
- **Adjusted association**, `GA`-confounded, not causal (above).
- Reporting-tier fits (6 chains × 6000); `gamma_blocks` is stable dev → reporting.

## Follow-ups (planned)

- **Reporting-tier GB fit:** the GB corroboration above is dev-tier (noisy ranks, no SHAP direction); a reporting-tier fit (cv = 51 + SHAP) would sharpen the per-outcome ordering and give a signed direction.
- **Byrne cohort** (Phase 3): the analogue `basmat` → `bpvs` (receptive), gated on the #164 Byrne data decisions and needing a covariate-adjusted vehicle on the historical panel (`basmat` is wave-3+; Byrne has no expressive-vocabulary measure).
