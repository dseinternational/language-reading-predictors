> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Notes and decision records

Notes record what was known or decided on their stated date. They are not all current instructions. Use [METHODS.md](../METHODS.md), the [model catalogue](../docs/models/README.md) and the [refit runbook](../docs/runbooks/full-statistical-model-refit.md) for current practice.

## Findings and rebuilds

The [1 September statistical findings](202609011800-findings-00-overview.md) link to 21 family summaries. The [boosting review](202609012030-gb-findings-review.md) covers the same date. These are the most recent full narrative findings series retained here, but they are **dated snapshots**, not a fresh evaluation of the present code.

The [8 September rebuild](202609080119-full-rebuild-both-layers.md) is a later execution record. The [21 September refit](20260921-full-statistical-refit.md) repeats the statistical layer only and records three declared acceptance-target remediations. The [16 September integrity fixes](20260916-codebase-review-and-run-integrity.md) changed fit provenance, resumption and bootstrap importance. Check a stored fit's configuration and current publication decision before reusing a numerical finding. This documentation review did not refit models or revalidate historical numbers.

The [August findings by question](202608182200-findings-by-question.md) retains cross-model questions and author decisions. Its numerical evidence is historical. The [September report plan](202609071300-technical-report-plan-v2.md) remains a proposal; its pending author decisions have not been approved by this cleanup.

## Decisions that govern current work

| Topic                        | Decision record                                                                                                                                                                                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Causal graph                 | [Team revision](202607101100-dag-revision-team-decisions.md), [RLI time-lagged graph](202607131200-time-lagged-dag.md) and [Byrne time-lagged graph](202608141700-byrne-lagged-dag-decision.md)                                                  |
| Credible intervals           | [Median, inner 50% and outer 89% intervals](202607172359-credible-interval-standard.md)                                                                                                                                                          |
| Practical differences        | [Standing threshold rule](202608191130-practical-difference-rule-confirmed.md) and [Action Picture Test thresholds](202608182015-apt-delta-threshold-ratification.md)                                                                            |
| Divergences                  | [Trace- and estimand-specific qualification policy](202608021625-divergence-qualification-policy.md)                                                                                                                                             |
| Phoneme-blending comparisons | [Scope](202608242000-blending-guessing-floor-scope-608.md), [evidence requirements](202608252100-blending-pair-binding-608-decision-2.md) and [gain-model exemptions](202608251100-gain-blending-guessing-floor-596.md)                          |
| Gain-factor interpretation   | [Primary and moderation specifications](202608071500-gf-391-findings-2-3-respec.md) and [later corrections](202608261200-gain-factors-575-decisions.md)                                                                                          |
| Crossover contrasts          | [t2 arm-gap estimand](202608241100-did-t2-estimand-signoff.md) and [later randomised schedule contrasts](202608262110-did-lf-estimand-label-sync-631.md)                                                                                         |
| Mediation                    | [Common baseline adjustment](202608231500-mediation-585-remediation.md)                                                                                                                                                                          |
| Prediction for new children  | [Joint-family prediction target](202609011600-joint-new-child-prediction-target-626.md)                                                                                                                                                          |
| Priors and release           | [Prior descriptors](202608311600-prior-descriptor-findings-637.md) and [missing prior-evidence policy](202608311200-prior-evidence-release-policy-637.md)                                                                                        |
| Data provenance              | [Trial archive](202609071500-incorporate-deposited-trial-archive.md), [Byrne source reconciliation](202608161340-byrne-source-provenance-reconciliation.md) and [word-repetition quarantine](202608262120-erb-word-repetition-quarantine-631.md) |
| Assessment intervals         | [Unequal intervals, the t2 timing bound and `lcsm-167`](202609212100-assessment-interval-lengths.md)                                                                                                                                             |
| Retired analysis             | [Pooled-moderation command](20260912-pooled-moderation-retirement.md)                                                                                                                                                                            |

Other retained design notes, source audits and reviews provide the evidence behind these decisions. A proposal is not an approved method. A completed audit's list of open issues describes its date; check later remediation notes and current code before treating those issues as unresolved.

## Earlier findings snapshots

The 17 September cleanup removed repeated family findings from 16, 20 and 21 July, 5 August and 18 August, together with superseded June-to-August run summaries, the retired feature-selection table, six completed pipeline-migration logs, two superseded report-layout notes and the retired pooled-moderation analysis. The current architecture guide replaces those migration instructions. The September series and current guides replace their role as reading material. Scientific decisions, source audits and distinctive methodological reviews remain.

The removed files remain available in [the notes directory before this cleanup](https://github.com/dseinternational/language-reading-predictors/tree/b62fd6ce093c2cb81e5c842cf96ff11ffa6e4d95/notes). Citations from retained notes point to that exact version, preserving the evidence cited at the time. To recover a file locally, use the recorded commit and its original path, for example:

```bash
git show b62fd6ce093c2cb81e5c842cf96ff11ffa6e4d95:notes/202607161800-findings-gain_factors.md
```

Use those files only to reconstruct the earlier analysis. Their intervals, causal labels, fitted samples and release rules may have been superseded.
