> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).
>
> Boosting-refit link added by a LLM-based AI tool (Claude Code/Opus 5).
>
> September-rebuild link added by a LLM-based AI tool (Claude Code/Opus 5.5).

# Notes and decision records

Notes record what was known or decided on their stated date. They are not all current instructions. Use [METHODS.md](../METHODS.md), the [model catalogue](../docs/models/README.md) and the [refit runbook](../docs/runbooks/full-statistical-model-refit.md) for current practice.

## Findings and rebuilds

Choose the record for the question you need to answer:

- [1 October rebuild](20261001-full-rebuild-both-layers.md) records the latest full rebuild retained here. It includes the Windows Numba workaround, convergence remediation and comparisons with the preceding run.
- [4 October corrections](20261004-statistical-review-corrections.md) changed mediation sensitivity, pooled-level row identity, interpretation and new-child K-fold validation after that rebuild. A stored result must meet the current requirements before reuse.
- [September statistical findings](202609011800-findings-00-overview.md) and [boosting findings](202609012030-gb-findings-review.md) are the latest full narrative series, but their numbers describe older fits. The boosting series used MAE. The [Huber decision and retune](202609221800-gb-huber-retune-refit.md) changed that policy.
- [August findings by question](202608182200-findings-by-question.md) retains cross-model questions and author decisions. Its numerical evidence is historical.
- [September report plan](202609071300-technical-report-plan-v2.md) remains a proposal. This cleanup does not approve its pending author decisions.

Earlier rebuilds remain evidence of their own runs. Check each fit's configuration, data and environment identities, diagnostics and current publication decision before citing a numerical result. The presence of a trace, table or report page does not establish that the result remains eligible for publication.

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
| Prediction for new children  | [Prediction target](202609011600-joint-new-child-prediction-target-626.md), [PSIS validation](20260923-statistical-review-corrections.md) and [K-fold batch validation](20261004-statistical-review-corrections.md)                              |
| Priors and release           | [Prior descriptors](202608311600-prior-descriptor-findings-637.md) and [missing prior-evidence policy](202608311200-prior-evidence-release-policy-637.md)                                                                                        |
| Data provenance              | [Trial archive](202609071500-incorporate-deposited-trial-archive.md), [Byrne source reconciliation](202608161340-byrne-source-provenance-reconciliation.md) and [word-repetition quarantine](202608262120-erb-word-repetition-quarantine-631.md) |
| Assessment intervals         | [Unequal intervals, the t2 timing bound and `lcsm-167`](202609212100-assessment-interval-lengths.md)                                                                                                                                             |
| Retired analysis             | [Pooled-moderation command](20260912-pooled-moderation-retirement.md)                                                                                                                                                                            |

Other retained design notes, source audits and reviews provide the evidence behind these decisions. A proposal is not an approved method. A completed audit's list of open issues describes its date; check later remediation notes and current code before treating those issues as unresolved.

## Removed or superseded records

The [7 October documentation review](20261007-documentation-review.md) lists the removed migration plans, completed repair logs, old tuning records and duplicate review notes. Their exact versions remain in Git history. References to their evidence must point to those versions rather than to a newer document with different results.

The 17 September cleanup removed earlier repeated findings and run summaries. Those files remain in [the notes directory before that cleanup](https://github.com/dseinternational/language-reading-predictors/tree/b62fd6ce093c2cb81e5c842cf96ff11ffa6e4d95/notes). To recover a historical file locally, name its commit and original path:

```bash
git show b62fd6ce093c2cb81e5c842cf96ff11ffa6e4d95:notes/202607161800-findings-gain_factors.md
```

Use removed files to reconstruct the earlier analysis. Their intervals, causal labels, samples and release rules can have been superseded.
