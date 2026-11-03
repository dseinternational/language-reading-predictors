> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Dependency update, 3 November 2026

The Python lockfile advances 19 packages and adds `better-optimize`, required by PyMC 6.4.0. The shared research library remains at `dse-research-utils` 0.19.3, its latest release. Its selected extras, dependency bounds and the supported Python 3.14 platforms remain unchanged.

| Package            | Previous version | Updated version |
| ------------------ | ---------------- | --------------- |
| anyio              | 4.15.1           | 4.16.0          |
| azure-storage-blob | 12.31.0          | 12.31.1         |
| better-optimize    | New dependency   | 0.5.1           |
| fastjsonschema     | 2.22.2           | 2.22.3          |
| filelock           | 4.2.4            | 4.2.7           |
| ipython            | 9.17.1           | 9.18.1          |
| jsonpointer        | 3.6              | 3.8             |
| jupyterlab         | 4.6.4            | 4.6.5           |
| mypy               | 2.4.0            | 2.4.1           |
| mypy-extensions    | 1.1.0            | 1.2.0           |
| notebook           | 7.6.3            | 7.6.4           |
| pandas-stubs       | 3.0.5.260914     | 3.1.0.261029    |
| platformdirs       | 4.13.7           | 4.14.3          |
| pyarrow            | 26.0.0           | 26.0.1          |
| pymc               | 6.3.2            | 6.4.0           |
| pytensor           | 3.3.4            | 3.3.7           |
| python-dateutil    | 3.0.0            | 4.0.0           |
| pywinpty           | 3.1.1            | 3.1.2           |
| soupsieve          | 3.0.4            | 4.0             |
| wrapt              | 2.5.1            | 2.5.2           |

The pnpm pin advances from 12.15.0 to [12.16.1](https://github.com/pnpm/pnpm/releases/tag/v12.16.1). Its lockfile also updates ansi-regex, fast-equals and semver. CSpell 10.4.0 and Prettier 3.9.10 remain current. CI advances [pnpm setup to 6.1.1](https://github.com/pnpm/action-setup/releases/tag/v6.1.1) and [uv setup to 10.3.3](https://github.com/astral-sh/setup-uv/releases/tag/v10.3.3). Checkout 7.0.1, Node setup 7.1.0, Hatchling 1.33.0 and the copied design tokens at `tokens/2026.3.7` are current.

The resolver retains Numba 0.68.0 because PyTensor requires `numba<=0.68.0`, and Cachetools 6.2.6 because PyMC requires `cachetools<7`. The available 0.68.1 and 7.2.8 releases are therefore excluded. No overrides weaken those bounds.

## Compatibility review

[PyMC 6.4.0](https://github.com/pymc-devs/pymc/releases/tag/v6.4.0) includes a substantive [LKJ correlation-prior correction](https://github.com/pymc-devs/pymc/pull/8497). LKJ priors describe uncertainty about correlations before observing the data. The correction changes prior draws for three or more correlated quantities, adds a missing change-of-variables term to `LKJCorr`'s density and corrects density normalisation. For three or more quantities, the density correction can change fitted results even when the prior's concentration setting is fixed. `LKJCholeskyCov` has corrected prior draws and density normalisation; its density shape with fixed concentration was already correct.

This repository uses those distributions in its joint, historical and correlated-factor model factories. The update retains the factories and their declared priors. A regression check compares every pair's mean squared correlation in two- and three-outcome PyMC draws with the repository's prior-spread formula. The tolerance allows six sampling standard errors. The previous test comment describing unequal pair spreads is removed because PyMC has fixed that error.

Saved results are not validated by an environment update or a passing test suite. Existing fits and their recorded environments remain intact. Affected `LKJCorr` models with three or more quantities need refitting before their results can be described as using the corrected prior. Recomputing diagnostics alone cannot correct their stored posterior draws. Retain each historical fit's own prior draws when describing that fit. No study model fits are rerun in this update.

[PyTensor 3.3.5](https://github.com/pymc-devs/pytensor/releases/tag/rel-3.3.5), [3.3.6](https://github.com/pymc-devs/pytensor/releases/tag/rel-3.3.6) and [3.3.7](https://github.com/pymc-devs/pytensor/releases/tag/rel-3.3.7) include graph and Numba changes. The existing suite exercises model construction, prior prediction and a real PyMC/nutpie sampling check.

[Dateutil 4.0](https://github.com/dateutil/dateutil/releases/tag/4.0.0) drops Python versions below 3.11 and rejects malformed numeric fields in ISO dates. Python 3.14 meets its requirement, and repository source and scripts contain no direct dateutil calls. [Soup Sieve 4.0](https://github.com/facelessuser/soupsieve/releases/tag/4.0) changes HTML template and iframe selection rules. The repository has no direct calls to it. Notebook HTML export is checked because notebook conversion uses that dependency indirectly.

## Validation

Frozen Python and Node installs, Python dependency compatibility, the offline lockfile check, Ruff, mypy across 537 source files, registry documentation checks, theme-integrity checks, Markdown formatting and spelling pass. Installed Python package versions match the lock. The three focused LKJ checks pass. The study data's 216 rows and 146 columns survive an Arrow round trip with identical values and types, and notebook HTML export passes. Source and wheel builds pass; archive inspection confirms that local environments and the uv cache are excluded. See the PR for the complete test-suite result. No stored scientific results are regenerated.
