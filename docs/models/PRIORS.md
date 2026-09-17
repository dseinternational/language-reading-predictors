> [!NOTE]
> Substantially revised by a LLM-based AI tool (Codex/GPT-6).
>
> Drafted by a LLM-based AI tool (Codex/GPT-5).
>
> Available-case modified ITT terminology updated by a LLM-based AI tool (Codex/GPT-5).

# Priors and their rationale

A prior assigns probabilities to possible parameter values before using the observed outcomes. Its meaning depends on the measurement scale, the model equation and the role of the parameter. A small coefficient on a logit scale can still imply a large change in a score with many possible items.

## Find the prior used in a fit

Use the fit's `priors_table.csv`, prior density panels and `config.json`. The table describes the variables in the model that was built, including parameter overrides. A table of shared defaults cannot replace that record.

[priors.py](../../src/language_reading_predictors/statistical_models/priors.py) defines shared constructors and `PriorDescriptor`. Each factory records a descriptor when it creates a free random variable. It names the constructor, fitted distribution, scientific role, rationale, density panel and source of the declaration.

- Named constructors record their defaults through `.to_pymc(...)`. The factory supplies `role` and `rationale` when their meaning differs in that model.
- Inline priors use `priors.declare(...)` at the construction site.
- `EXTERNAL_PRIORS` explicitly describes variables created by the shared HSGP library.
- A missing descriptor stops the fit. The report must not infer a prior's meaning from a variable name.

The former June inventory and name-based lookup instructions have been removed. They predated the [descriptor decision](../../notes/202608311600-prior-descriptor-findings-637.md) and several prior changes. [The prior-inventory tests](../../tests/statistical_models/test_prior_inventory.py) check coverage against constructed models.

## Choose and check priors

Use measurement limits, external evidence, stated expert judgement and justified model constraints. Record the source and timing of each choice. A choice made after inspecting these outcomes must be labelled as such. Computational diagnostics can motivate a change, but they do not supply independent scientific evidence for it.

Check a prior on the scale of the reported quantity. Draw parameters from the prior, calculate the same contrast used for the findings, then inspect its distribution in probabilities or score units. Use the model's score link and `MEASURES[symbol].n_trials`; a guessing-floor link and an ordinary logit link imply different score distributions even with identical coefficient priors.

Prior-predictive checks also simulate observations. Check score ranges, zero and ceiling frequencies, dispersion and differences between arms or baseline groups. A Beta-Binomial respects bounded scores and allows extra variation. It does not prove that the test items have equal difficulty or that stopping rules are irrelevant. Persistent mismatch calls for a likelihood sensitivity analysis.

## Treatment and baseline priors

The default single-outcome treatment prior has logit-scale standard deviation 0.5 for proximal outcomes and 0.3 for the distal set `R`, `E`, `T`, `F`, `UR` and `UE`. [The shared resolver](../../src/language_reading_predictors/statistical_models/factories/base.py) supplies these defaults; a declared sensitivity may override them. Joint models use their own declared treatment prior. Read the fitted table before assuming a default applies.

The distal tier was adopted after the trial's posterior results were available. It is a post-data regularisation decision. Stability under later sensitivity analyses does not make it prospectively specified. The [original audit](../../notes/202607011600-issue-141-prior-audit.md) records that timeline.

Own baseline and age can improve precision in the randomised contrast. They do not create its causal identification. Their equations and priors can still change the estimate in a small sample. Check the unadjusted arm contrast and the registered baseline-prior sensitivities alongside the adjusted result.

For floored phonetic spelling and nonword reading, the headline is a difference in the probability of moving off zero. Its Bernoulli model needs its own prior checks. If power scaling flags treatment-prior conflict, the required grid varies the treatment-prior standard deviation over 0.5, 1.0 and 1.5, each with and without linear age. Every cell must meet the convergence and trace-provenance requirements. Report the range of estimates across the grid; completion alone does not establish insensitivity.

## Scientific roles

| Role          | Meaning                                                                                                                                           |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `causal`      | A contrast whose causal interpretation follows from the design and stated assumptions.                                                            |
| `regime`      | A randomised early-start versus delayed-start treatment-schedule contrast. It does not isolate a treatment mechanism.                             |
| `precision`   | A pre-treatment covariate included to explain outcome variation and improve precision. Its specification can still affect the treatment estimate. |
| `association` | A conditional relationship that may remain confounded.                                                                                            |
| `nuisance`    | A supporting term, such as an intercept, dispersion or child-level scale.                                                                         |
| `gp`          | A component of a Gaussian-process function, assessed through the implied curve.                                                                   |

A constructor's default role need not be its role in every model. For example, a group coefficient in an onset-aligned model is an association because intervention onset was not randomised across the aligned windows.

## Read the sensitivity evidence

Compare the named estimand across the registered prior choices. Keep the fitted rows, model terms and comparison scale aligned. Inspect convergence before interpreting movement. The [refit runbook](../runbooks/full-statistical-model-refit.md) gives the commands and required evidence bundles.

Power scaling asks how modest changes in prior or likelihood weight affect the posterior. A warning identifies a quantity to investigate; an unflagged parameter is not proof that all plausible priors agree. Missing estimand-scale prior evidence qualifies a result under the [release policy](../../notes/202608311200-prior-evidence-release-policy-637.md). Other failed release requirements can still withhold it.

Report the median, inner 50% and outer 89% equal-tailed credible intervals, and the probability of the named claim. Explain any material dependence on prior choice in score units or probabilities. See [METHODS.md](../../METHODS.md#reporting-results) for the full reporting rules.
