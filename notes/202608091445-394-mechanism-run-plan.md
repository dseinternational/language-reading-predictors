> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Mechanism typed settings and run-plan boundary

## Decision

Issue #394's next typed-plan tranche is the `mechanism` family. This ordering supersedes the earlier suggestion to take `joint` next because open issue #433 needs new mechanism likelihood and exposure-transform settings, and those scientific capabilities need a validated family boundary before they are introduced. The structural and scientific changes remain separate: this tranche records and validates the existing mechanism design only; it does not add an off-floor likelihood, a `log1p` exposure transform, a no-exposure comparator, a new model ID or any new result.

## Boundary

`MechanismModelSettings` is the immutable declaration. `resolve_mechanism_run_plan()` strictly translates the legacy `spec.extra` surface, rejects unknown keys, wrong types and contradictory combinations, and produces a pure `MechanismRunPlan` before the output context or data loader is touched. The plan owns the requested outcomes, the loader's requirement for every loaded outcome's period-start score, autoregressive baseline, raw-covariate adjustment and complete-case declarations, age and subject-effect choices, exposure alignment, moderation, linear-versus-HSGP form, thin-support HSGP settings and items reference quantiles. It validates bounded symbols against the measure registry and requires an explicit missingness-indicator or complete-case policy for every filled covariate; filled exposures and moderators must be genuinely observed. It also records the current fixed Beta-Binomial likelihood and `y_post` observation node plus plain-language design, estimand, causal status, analysis population and missing-data qualification.

The pre-existing `MechanismPlan` remains the post-load fit/refit construction object. It now consumes the validated run plan and records the effective `adjust_for` set after preprocessing drops constant covariates. The primary fit and exact-LOO refits therefore continue to share one factory-keyword mapping, including data-derived effective adjustment, while `config.json` and `model_recipe.md` preserve the complete declared contract.

## Behaviour-preservation rules

Every registered mechanism specification remains on the legacy declaration route for this tranche, but the legacy keys are a closed, validated set rather than an open dictionary. The global `target_accept` key is accepted but remains owned by the shared sampler precedence; it is not misclassified as a scientific model setting. The mechanism pipeline no longer reads `spec.extra` directly. Existing models retain the same prepared rows, measure/covariate timing, factory arguments, likelihood, priors, free variables, diagnostic variables, sampling configuration and artefact schemas. The only intended fit-output additions are the generated `model_recipe.md` and `resolved_run_plan` metadata.

## #433 remains a separate methodological decision

The proposed word-reading to nonword-decoding registration is not ready to be folded into this structural change. The `18 ± 4.4` ELPD cost for dropping the exposure and the winning log shape reported in #433 came from a graded Beta-Binomial probe on transitions with low baseline nonword scores, whereas the proposed primary is a Bernoulli model that has not yet been compared against its null. The issue must also choose between true floor exit among baseline-floor transitions and off-floor prevalence over all transitions, settle baseline-status handling, pre/post covariate timing, the on-intervention term, slope prior, graded-secondary population and a genuine no-exposure comparator. Those choices change the estimand or fitted equation and require their own dated method decision and PR.

## Verification contract

Tests cover strict settings validation, typed/legacy parity, pre-I/O failure, covariate-exposure and covariate-moderator loading, effective factory arguments, diagnostics, recipe/reporting dispatch, exact-refit compatibility and all 34 registered mechanism specifications. A source guard prevents direct `spec.extra` reads from returning to `pipelines/mechanism.py`. The incremental MyPy gate expands to `mechanism.py`; focused mechanism/factory/refit tests, the full Python suite, Ruff, MyPy, Markdown formatting and spelling must pass before merge.
