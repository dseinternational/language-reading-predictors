> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Research utilities 0.15.0: lifting the NumPy ceiling, and Optuna 5.0

[Issue #667](https://github.com/dseinternational/language-reading-predictors/issues/667) moves the `dse-research-utils` pin to tag `v0.15.0`, which resolves to commit `e818caf52c0a73aed4ba5286bdfdc58c7867ea69`. The release changes no library API. It raises the shared dependency floors this repository inherits transitively and lifts the NumPy ceiling that has capped the stack since 0.11.0, so the work that needs judgement is entirely downstream: whether the new numerical stack changes any model, and what Optuna 5.0 does to the tuner. The extras are unchanged. The upstream account is the [0.15.0 migration guide](https://github.com/dseinternational/research/blob/v0.15.0/docs/migrating-to-0.15.md).

## A lifted ceiling does not move a pinned version

The issue's instruction — change the tag, then `uv lock` and `uv sync --locked` — does not produce the stack it describes. `uv lock` is conservative: it re-resolves only what the changed constraints force, and a _widened_ ceiling forces nothing. Running it moved five packages and left `numpy` at 2.4.6, `numba` at 0.66.0 and `pymc` at 6.3.1 — that is, the entire point of the release inert, with the repository declaring a `<2.6` ceiling while continuing to install 2.4.6.

The three packages in that chain were therefore upgraded explicitly:

```bash
uv lock --upgrade-package numpy --upgrade-package numba --upgrade-package pymc
```

which yields exactly the stack the issue names: numpy 2.5.3, numba 0.67.0, pytensor 3.3.1, pymc 6.3.2. A targeted upgrade rather than a bare `uv lock --upgrade`, so this change stays the dependency move it claims to be and does not silently absorb the weekly Dependabot sweep.

## What actually moved here

Eleven packages, with nothing added or removed:

| Package                  | Before       | After        |
| ------------------------ | ------------ | ------------ |
| `dse-research-utils`     | 0.14.0       | 0.15.0       |
| `numpy`                  | 2.4.6        | 2.5.3        |
| `numba`                  | 0.66.0       | 0.67.0       |
| `llvmlite`               | 0.48.0       | 0.49.0       |
| `pytensor`               | 3.3.0        | 3.3.1        |
| `pytensor-distributions` | 0.2.0        | 0.3.2        |
| `pymc`                   | 6.3.1        | 6.3.2        |
| `preliz`                 | 0.27.1       | 0.28.0       |
| `optuna`                 | 4.9.0        | 5.0.0        |
| `optuna-integration`     | 4.9.0        | 5.0.0        |
| `numpy-typing-compat`    | 20260602.2.4 | 20260602.2.5 |

The upstream release notes list eight further raised floors — `statsmodels` 0.15.0, `arviz-stats` 1.3.2, `arviz-plots` 1.3.1, `xgboost` 3.4.1, `polars` 1.44.1, `pyreadstat` 1.3.6, `orjson` 3.12.0, `seaborn` 0.13.2 and `networkx` 3.6.1 — but **every one of them was already locked at or above the new floor here**, having arrived through Dependabot's weekly sweep. The floor raise codified what this repository was already installing, and none of those versions changed in this commit. In particular the migration guide's instruction to "check `statsmodels` 0.15.0 against any regression output the project reports" has no work behind it for this repository: the two scripts that import `statsmodels.formula.api` (`scripts/within_child_interaction_check.py` and `scripts/byrne_nonverbal_vocab_diagnostic.py`) have been running on 0.15.0 since before this change.

The real numerical delta is therefore narrower than the issue's table suggests, and it is the part that matters: NumPy, numba/llvmlite, PyTensor and PyMC — the compiled core every posterior is sampled through.

## No model changes under the new stack

`model_design_identity` hashes the built PyTensor graph (`structure_sha256`) and the shared-variable data behind it (`design_sha256`). Both were recomputed for **every one of the 276 registered statistical models** under the old stack and the new one, and compared.

The usual three-way comparison — `main` with the old library, `main` with the new, the branch with the new — collapses to two legs here, because this branch changes no Python that participates in building a model. Its only code change is three type annotations that mypy requires (below), and those alter no runtime value. The two legs are `main` at the 0.14.0 lockfile, in a separate worktree with its own environment, and this branch at the 0.15.0 lockfile.

**All 276 models agree exactly on both digests.** Nothing about the graph or the data behind it moves under NumPy 2.5.3, numba 0.67.0, PyTensor 3.3.1 and PyMC 6.3.2. No fit needs resampling on account of #667.

The identities were recovered without sampling: `SharedFitStages.attach_built` is the one hook every family pipeline passes through after building its model and before any sampling, so patching it to capture the model and abort covers all twenty-three families with no per-family rebuild code. Fourteen models — the thirteen `concurrent` fits (`lrp-rli-ca-001` to `011`, `lrp-rli-ca-307` and `lrp-rlm-ca-002`) and the `joint_mechanism` fit `lrp-rli-jm-001` — interleave sub-fits ahead of the primary attach, which is the documented concurrent-wave exception, so those were run a second time with sampling allowed and captured at the same hook.

## Posterior draws are not bit-reproducible across the stacks

An unchanged model is not the same claim as an unchanged posterior sample, and the two come apart here. `tests/statistical_models/test_joint_dependence.py` fits three coarse simulated joint models — 80 synthetic children, 500 draws, two chains, `random_seed=41` — at simulated residual correlations of -0.7, 0 and +0.7. Under the new stack the `rho = -0.7` and `rho = +0.7` fits reproduce the old stack's summaries to six decimal places, while the `rho = 0` fit does not: its average-marginal-effect correlation moves from -0.035 to +0.113, and the test's `abs=0.1` tolerance around zero fails.

That pattern — two of three fits bit-identical, the third displaced — is the signature of a seeded NUTS trajectory diverging. A last-bit difference in the compiled log density changes one accept or U-turn decision, and from there the chain follows a different but equally valid path. It shows up in the `rho = 0` fit because that is the one whose residual block is weakly identified, so there is least in the posterior geometry to pull the two paths back together. Seed 43 is bit-identical on both stacks; seeds 41, 44, 45 and others are not.

Repeating the `rho = 0` fit across eight sampler seeds measures how much of this is ordinary Monte Carlo noise. Under the pre-#667 stack the estimate spans -0.035 to +0.042 (mean +0.007, sd 0.025); under the new one it spans -0.016 to +0.113 (mean +0.046, sd 0.048). So the spread roughly doubles and shifts slightly positive on this deliberately weakly-identified fixture. The `abs=0.1` tolerance was about four standard deviations on the old stack and about one on the new — it was calibrated against one stack's value rather than against the estimator's variability, which is why a change that moves no model could cross it.

The fix is to buy the precision rather than to widen the threshold: the fixture now samples **4,000 draws** and the tolerance stays at `abs=0.1`. Repeating the `rho = 0` fit over six seeds at each draw count shows why that is the right lever, and settles what the displacement was:

| draws | mean       | sd        | max \|r\| |
| ----- | ---------- | --------- | --------- |
| 500   | +0.031     | 0.051     | 0.113     |
| 1,000 | +0.027     | 0.039     | 0.063     |
| 2,000 | +0.010     | 0.035     | 0.051     |
| 4,000 | **+0.011** | **0.026** | **0.042** |

The spread falls about as 1/sqrt(draws), and — the part that matters — the estimate converges on +0.011 rather than on the +0.113 that seed 41 produced at 500 draws. The displacement was Monte Carlo noise in a quantity the fixture was not sampling hard enough to assert about, not a shifted posterior. At 4,000 draws the seed-to-seed sd is 0.026, which is what the pre-#667 stack had at 500, so `abs=0.1` is again the roughly four-deviation margin it was originally written as, and the widest of the six seeds sits at 0.042.

The cost is about five seconds. Compiling the model dominates this fixture, so eight times the draws takes the file from roughly 38 to 43 seconds. The directional ordering assertion, which the fixture's docstring names as the actual question, passes on every seed on both stacks either way; what the extra draws buy is the independence check beside it.

Nothing here bears on the registered fits, which sample at reporting resolution behind an ESS and R-hat gate rather than at 500 draws with no gate at all. The practical consequence is narrower and already implied by the identity result: the same model and the same seed do not have to yield the same draws once the compiled numerics change, so a posterior may only be compared with another sampled under the same environment lock. Which is exactly what the next section enforces.

## Every stored fit now refuses trace reuse

`environment_lock_sha256` digests the installed environment snapshot, and it is a bound field of the reuse contract. It moves with this upgrade, so all 276 stored fits now mismatch it: `fit_statistical_model.py --reuse-trace` is refused for every one of them until it is refitted.

This is the guard behaving as designed rather than a defect, and — given the identity result above — it is now known to be conservative here: the models are unchanged, so the stored posteriors remain the posteriors of the models the code builds today. What it costs is the ability to backfill figure or template changes onto the existing corpus without NUTS. **Whether to accept that or to schedule a rebuild is a decision for the maintainer, not a consequence of this PR**, and nothing here should be read as qualifying the stored results.

## Optuna 5.0 changes what the tuner searches

`scripts/tune_model.py` is the only consumer of the `tuning` extra. It uses none of what 5.0 removed — no `optuna.multi_objective`, no `constraints_func`, no `RDBStorage` or `JournalStorage` — and it creates a single-objective, in-memory, seeded study, so it runs unmodified. The behaviour underneath it does change:

- `TPESampler(multivariate=...)` now defaults to `None`, which for a single-objective study resolves to **True**. The tuner's searches were univariate TPE under 4.9 and are multivariate TPE under 5.0.
- `constant_liar` now defaults to `True`. This is a no-op here: it only accounts for concurrently running trials, and `study.optimize` is called sequentially.

The consequence is reproducibility, not correctness. The same seed no longer retraces the same search: running an identical 60-trial study under both versions gives different trial parameters from early on and a different best point (best value 0.0131 under 4.9, 0.0101 under 5.0). Several gradient-boosting model modules carry promoted hyperparameters whose provenance is recorded as a recipe — `lrp_rli_gbg_002` and `lrp_rli_gbg_004`, for instance, record "MAE-tuned by Optuna on the full predictor set (150 trials, seed 47; #169)". **That recipe no longer reproduces its result.** The promoted values are unaffected and remain what they were selected to be; what is lost is the ability to re-derive them by re-running the recorded command.

No stored study is at risk, because `output/tuning/` is empty and is not tracked. The follow-up worth considering separately is recording the Optuna version and resolved sampler settings in `study_summary.json`, so a future study is self-describing rather than relying on a version-sensitive seed. That was left out of this PR deliberately: changing what the tuner records, or pinning `multivariate=False` to preserve the old search, is a decision about the tuning method rather than part of a dependency bump.

## Type checking

NumPy 2.5's stubs no longer let mypy infer the element type of `np.asarray(<list of int>, dtype=int)`, which surfaced three `var-annotated` errors in two modules that were previously clean. They are fixed at the call sites — `positions` in `new_child_kfold.py`, `used_children` and `columns` in `lcf_inference.py` — annotated `np.ndarray` to match the surrounding convention. Neither module was added to the `ignore_errors` exemption list: that list can only shrink, and `tests/test_type_coverage.py` would have failed either way.

## A pre-existing disagreement, not caused by this upgrade

While comparing, the recomputed identities were also checked against the `model_design_identity` each stored fit recorded during the 2026-09-08 rebuild. 254 of the 262 comparable models match. Eight do not: `lrp-rli-itt-215`, `lrp-rli-itt-216`, `lrp-rli-itt-315`, `lrp-rli-jm-002`, `lrp-rli-lcf-001`, `lrp-rlm-jc-002`, `lrp-rlm-jc-102` and `lrp-rlm-mm-001`. In every case `design_sha256` matches and only `structure_sha256` differs, and the recomputed value is **identical under both library versions** — so the disagreement is between the stored artefacts and the current repository code, not between the two stacks. Each of the eight builds exactly one model, so this is not an artefact of capturing the wrong sub-fit.

This predates #667 and is out of its scope. It is recorded here because it is the same class of gap as the 47 mechanism identities noted under #662, and because anyone repeating this comparison after the upgrade would otherwise be tempted to attribute it to the new stack.

## Validation

- `uv sync --locked` installed `dse-research-utils` 0.15.0 at the released commit `e818caf5`; installed metadata confirms both the tag and the commit.
- `uv run pytest` — 3,456 passed, 1 skipped, 0 failed. The upgrade surfaced two failures, both addressed above and neither silenced: the two `tests/test_type_coverage.py` failures were the three NumPy 2.5 annotation errors, fixed at their call sites rather than exempted, and `test_joint_dependence.py::test_the_declared_contrast_carries_the_fitted_dependence` was the sampler-path divergence, resolved by sampling the fixture at 4,000 draws rather than 500, with its tolerance left where it was.
- `ruff check src/`, `npm run format:check` and `npm run spellcheck` pass.
- `model_design_identity` recomputed for all 276 registered models on both stacks: no difference in `structure_sha256` or `design_sha256`.
- Optuna 4.9 and 5.0 compared directly on one seeded study to establish that the search, not merely the version, changed.

No fits were run for this change beyond the sub-fits the fourteen interleaving models sample before their primary attach, which were written to a scratch output root and discarded. No stored artefact was modified.
