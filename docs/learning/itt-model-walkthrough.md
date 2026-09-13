> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Follow one statistical model from data to a reported difference

This walkthrough uses `lrp-rli-itt-001`, the model for taught receptive vocabulary, block 1 (`TR`). The score counts correct responses out of 24 items. You will build the registered model with synthetic data, inspect its assumptions, fit it, check the computation and calculate an assigned-arm difference in items. The example uses simulated children and does not estimate the study's result.

Start with [the runnable script](../../scripts/learn_itt_model.py). The [report reading guide](../models/_partials/_reading_guide.qmd) explains the terms used in reports. [METHODS.md](../../METHODS.md) gives the study's design and reporting rules.

## Run the example

From the repository root, install the declared environment and inspect predictions made before fitting:

```bash
uv sync
uv run python scripts/learn_itt_model.py --prior-only
```

Open `output/learning/itt-001/priors.csv`, `prior_effect_summary.csv` and `prior_score_check.png`. Then fit the model:

```bash
uv run python scripts/learn_itt_model.py
```

The defaults generate 80 children and fit four chains, each with 1,000 tuning steps and 1,000 retained draws. The seed is fixed so that you can repeat the example. Small numerical differences can occur across software versions and platforms. Use `--seed 73` to explore another simulated sample, or `--output-dir output/learning/another-run` to keep its files separately. The usual project output-root setting also applies.

The script deliberately keeps the steps visible. It uses the production data loader, model builder and summary functions, but calls the sampler directly. A production fit also runs predictive validation, sensitivity checks, publication rules and artifact recording through [the ITT pipeline](../../src/language_reading_predictors/statistical_models/pipelines/itt.py) and [the shared stages](../../src/language_reading_predictors/statistical_models/stages.py). This learning script does not certify a study result for publication.

## State the question before choosing the code

The first period compares assignment to immediate intervention with assignment to the wait-list arm, before that arm starts treatment. The target is the difference in expected follow-up scores under those two assignments, averaged over the children included in the fit.

Randomisation supports a causal interpretation of assignment in the full randomised cohort. The real analysis begins with 54 archived children and applies observed-data requirements. We therefore call its estimate an **available-case modified intention-to-treat estimate**. Adjusting for baseline score and age can improve precision; it does not repair selection caused by unavailable children or scores. The synthetic example has complete data and avoids that particular selection problem by construction.

Find the declaration in [lrp_rli_itt_001.py](../../src/language_reading_predictors/statistical_models/lrp_rli_itt_001.py). `SPEC` names the question and selects `IttModelSettings()`. The [ITT settings and resolver](../../src/language_reading_predictors/statistical_models/itt.py) turn that declaration into one validated plan. The plan supplies the loader and factory arguments, so they cannot silently use different versions of the specification.

## Read the prepared arrays

The synthetic CSV has two rows per child, one at baseline and one at follow-up. [The loader](../../src/language_reading_predictors/statistical_models/preprocessing.py) pairs those rows. The model receives one observation per child:

| Array               | Meaning                                                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `G`                 | Assignment, with 1 for immediate intervention and 0 for wait-list. The source CSV uses group codes 1 and 2; the loader converts them. |
| `post_counts["TR"]` | Correct responses at follow-up, an integer from 0 to 24.                                                                              |
| `pre_logit["TR"]`   | Baseline score transformed to a log-odds scale.                                                                                       |
| `A_std`             | Baseline age, centred and divided by its sample standard deviation.                                                                   |

The transformed baseline is `log((score + 0.5) / (24 - score + 0.5))`. Adding half an item to each side keeps zero and perfect scores finite. A baseline value of zero on this transformed scale means half the items were correct. An age value of zero means the sample's mean baseline age; an age value of one means one sample standard deviation above it.

Array alignment matters. The same index must refer to the same child in every array. After a builder removes rows, summaries must use `built.prepared.G`, which matches the model's observations.

## Connect the formula to the factory

In [factories/itt.py](../../src/language_reading_predictors/statistical_models/factories/itt.py), the default model builds a linear predictor for child $i$:

$$
\eta_i = \alpha + \tau G_i + \gamma_{\mathrm{own}} x_i + \gamma_A a_i,
\qquad \mu_i = \frac{1}{1+\exp(-\eta_i)}.
$$

Here $x_i$ is the transformed baseline and $a_i$ is standardised age. The inverse-logit function, called `expit` in Python, maps any real number to a probability between zero and one. The model's expected follow-up count is $24\mu_i$.

The observed count follows a Beta-Binomial distribution with 24 trials and shape parameters $\mu_i\kappa$ and $(1-\mu_i)\kappa$. One way to understand this is to draw a child's probability from a Beta distribution, then draw a count from a Binomial distribution with that probability. Smaller positive values of $\kappa$ allow more variation in scores around the same expected count. The likelihood assigns probabilities to possible observed scores given the model parameters.

The priors describe parameter values before using the follow-up outcomes. `Normal(mean, standard deviation)` describes a bell-shaped distribution; `HalfNormal(scale)` keeps only its positive half. In this model:

| Parameter   | Prior             | Role                                                                                                             |
| ----------- | ----------------- | ---------------------------------------------------------------------------------------------------------------- |
| `alpha`     | `Normal(0, 1.5)`  | Reference log odds when the other inputs are zero.                                                               |
| `tau`       | `Normal(0, 0.5)`  | Assigned-arm difference in log odds. Positive values favour immediate intervention.                              |
| `gamma_own` | `Normal(1, 0.25)` | Baseline-score coefficient. A value of one carries the baseline log odds into the follow-up predictor unchanged. |
| `gamma_A`   | `Normal(0, 0.3)`  | Baseline-age coefficient used for precision.                                                                     |
| `kappa`     | `HalfNormal(50)`  | Variation beyond that of a Binomial count with a fixed probability.                                              |

The script writes this table from the priors that the factory actually built. Read that generated table if the registered specification changes. These are modelling assumptions, not facts established by the data.

## Check assumptions before fitting

`pm.sample_prior_predictive` first draws parameter values from the priors, then draws scores using those values. `prior_score_check.png` compares their pooled score distribution with the synthetic observations. `prior_effect_summary.csv` shows the assigned-arm differences implied by the same prior draws.

Look for assumptions that permit implausible scores or differences. The count likelihood already respects the score bounds, so staying between 0 and 24 alone is a weak check. Check where the distribution places its mass, including zero and perfect scores. This pooled plot can conceal differences between arms or baseline levels; production checks also examine those groups. Change a prior in a separate exercise only after stating the assumption you intend to change.

## Check the fit before interpreting it

`pm.sample` combines the likelihood and priors to approximate their posterior distribution, the distribution of parameters conditional on the observed data and model. Each chain is a separate simulation. Retained draws within a chain remain dependent, so 4,000 retained draws do not mean 4,000 independent observations.

Read `sampling_diagnostics.csv` and the printed divergence count. R-hat compares the chains; values near one support agreement. Effective sample size, or ESS, estimates how much information the dependent simulation contains for a summary. Monte Carlo standard error, or MCSE, describes uncertainty from using a finite simulation. It is separate from the posterior uncertainty about a parameter. Divergences indicate that the sampling algorithm had difficulty exploring parts of the distribution.

The project's ordinary computation gate requires R-hat at most 1.01, bulk and tail ESS at least 400, no divergences and sufficient movement through the sampler's energy distribution, measured by BFMI. The teaching script prints a subset of these checks; production uses [convergence.py](../../src/language_reading_predictors/statistical_models/convergence.py). Passing computational checks does not establish that the model fits the data or identifies a causal effect.

Next inspect `posterior_score_check.png` and `posterior_predictive_checks.csv`. Posterior predictive draws simulate new scores using parameters drawn from the fitted posterior. The CSV compares observed score summaries with their distribution across replicated datasets. Its 95% prediction-check intervals have a separate purpose from the project's 89% posterior intervals for estimated quantities.

## Calculate a difference within every draw

`tau` is a log-odds coefficient. Its numerical value is not an items difference. For each posterior draw, the summary removes the fitted assignment contribution from every child's predictor, then evaluates both assignments at that child's baseline score and age:

```python
eta_without_assignment = eta - tau * group
probability_immediate = expit(eta_without_assignment + tau)
probability_waitlist = expit(eta_without_assignment)
effect_items = 24 * (probability_immediate - probability_waitlist).mean()
```

This code describes one draw, with one value per child in each probability array. Repeat the calculation across all draws, then take the median and interval of those differences. [summaries/itt.py](../../src/language_reading_predictors/statistical_models/summaries/itt.py) performs that calculation with explicit observation and sample axes. The script writes each items difference to `effect_draws.csv` and the summary to `treatment_summary.csv`.

The 89% equal-tailed interval leaves 5.5% of the posterior draws below its lower limit and 5.5% above its upper limit. Its interpretation is conditional on the model and data. The coverage is a project reporting convention, not a threshold that makes a result correct.

Do not substitute median coefficients before calculating the difference. For three joint draws with baseline logits `[-4, 0, 4]` and treatment coefficients `[1, 2, 3]`, the probability differences are approximately `[0.0294, 0.3808, 0.0171]`. Their median is `0.0294`. Using the median baseline and treatment coefficient gives `0.3808`. Multiplying a probability difference by a fixed item count preserves its median; combining several uncertain coefficients through a nonlinear function need not.

## Extend the example one question at a time

- Follow [the gain-factor family](../../src/language_reading_predictors/statistical_models/gain_factors.py) to see repeated transitions within children. Its child intercept represents partially pooled differences between children; it does not fully control latent ability. Its causal marginal uses the randomised first period, and predictive validation holds out children together.
- Follow [ITT-009](../../src/language_reading_predictors/statistical_models/lrp_rli_itt_009.py) to a heavily floored outcome. Check its declared likelihood and baseline rule before reusing a graded-score formula. A binary off-floor probability answers a different question from an expected item count.
- Follow [the mediation calculation](../../src/language_reading_predictors/statistical_models/mediation.py). It integrates over a mediator distribution within each posterior draw, using an exact sum for finite counts or checked quadrature for normal mediators. That integration has its own numerical error, separate from posterior sampling error. Causal interpretation also needs assumptions about mediator-outcome confounding beyond randomised assignment.

Documentation licensed under [CC BY 4.0](../LICENSE). The linked Python example carries the repository's source-code licence.
