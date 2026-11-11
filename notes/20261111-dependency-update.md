> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5.5).

# Dependency update, 11 November 2026

The shared research library advances from `dse-research-utils` 0.19.3 to 0.19.4 at commit `bd916566081579e19031cabb637475a26cbd62aa`. That release changes dependency minimums only; its public helper signatures are unchanged. The selected extras, dependency bounds and supported Python 3.14 platforms are unchanged. The Python lockfile advances 14 packages, and the Ruff minimum advances to 0.17.4.

| Package            | Previous version | Updated version |
| ------------------ | ---------------- | --------------- |
| dse-research-utils | 0.19.3           | 0.19.4          |
| filelock           | 4.2.7            | 4.2.10          |
| jsonpointer        | 3.8              | 3.10            |
| jupyterlab         | 4.6.5            | 4.6.6           |
| notebook           | 7.6.4            | 7.6.5           |
| pandas             | 3.1.0            | 3.1.1           |
| platformdirs       | 4.14.3           | 4.14.5          |
| pytensor           | 3.3.7            | 3.3.8           |
| pyzmq              | 27.2.0           | 27.3.0          |
| ruff               | 0.17.3           | 0.17.4          |
| soupsieve          | 4.0              | 4.1             |
| sqlalchemy         | 2.1.6            | 2.1.8           |
| threadpoolctl      | 3.7.0            | 3.8.0           |
| tornado            | 6.6              | 6.6.1           |

The pnpm pin advances from 12.16.1 to [12.18.0](https://github.com/pnpm/pnpm/releases/tag/v12.18.0) and CSpell from 10.4.0 to 10.4.1. The pnpm lockfile also refreshes the companies, data-science, common-misspellings, npm, Python and software-terms dictionaries, `ansi-regex` and `commander`. Checkout 7.0.1, Node setup 7.1.0, pnpm setup 6.1.1, uv setup 10.3.3, Hatchling 1.33.0 and the copied design tokens at `tokens/2026.3.7` are already current.

## Held-back releases

Releases less than one day old are held back. pnpm enforces a one-day minimum release age and rejected [Prettier 3.10.0](https://github.com/prettier/prettier/releases/tag/3.10.0), published a few hours before this update. Prettier therefore stays at 3.9.10. pnpm wrote a local exclusion that would bypass the policy; that exclusion is not committed. The Python lock applies the same cutoff, so h5netcdf 1.8.3, jsonpointer 3.12, markdown-it-py 4.3.0, mdit-py-plugins 0.7.0 and nvidia-nccl-cu13 2.32.6 wait for the next update. The lock was resolved with `--exclude-newer` set to that cutoff and then relocked without it, so `uv.lock` does not record the cutoff.

The resolver retains Numba 0.68.0 because PyTensor 3.3.8 still requires `numba<=0.68.0`, and Cachetools 6.2.6 because PyMC 6.4.0 requires `cachetools<7`. The newer Numba 0.68.1 and Cachetools 8.0.0 releases are therefore excluded. No overrides weaken those bounds. Python 3.15 is not adopted because PyTensor 3.3.8 requires Python below 3.15.

## Compatibility review

The [0.19.4 migration guide](https://github.com/dseinternational/research/blob/v0.19.4/docs/migrating-to-0.19.md) raises the library's minimums to versions this lockfile already used, apart from pandas 3.1.1. `docs/shared-utilities.md` now links to it.

[Pandas 3.1.1](https://pandas.pydata.org/docs/whatsnew/v3.1.1.html) fixes missing datetime values becoming Unix-epoch timestamps during conversion to Arrow timestamp types, data sharing after Arrow-backed forward and backward fill, and dateutil time-zone compatibility. The study data has no datetime columns. Loading the study data under pandas 3.1.0 and 3.1.1 produced the same 216 rows and 146 columns with an identical digest of values, data types, categories and missing-value counts. The loaded frame also survives an Arrow round trip with identical values and types.

[PyTensor 3.3.8](https://github.com/pymc-devs/pytensor/releases/tag/rel-3.3.8) fixes unweighted `choice` sampling in the Numba backend and changes Numba compilation. Repository model code does not call PyTensor's `choice`; the posterior-score simulation uses NumPy's generator. The compilation changes are exercised by the test suite's PyMC/nutpie sampling check and by a dev-configuration ITT fit (`lrp-rli-itt-007`) written to a scratch output directory.

[Tornado 6.6.1](https://www.tornadoweb.org/en/stable/releases/v6.6.1.html) includes security fixes, including one that stops Windows static-file handlers serving NTFS alternate data streams. Tornado, pyzmq, JupyterLab and Notebook are used only by the notebook environment. [Soup Sieve 4.1](https://github.com/facelessuser/soupsieve/releases/tag/4.1) adds thread safety and selector parsing fixes; notebook HTML export uses it indirectly. [threadpoolctl 3.8.0](https://github.com/joblib/threadpoolctl/blob/master/CHANGES.md) reduces Linux overhead and changes OpenBLAS thread-pool control; scikit-learn uses it to manage native thread pools.

Saved results are not validated by an environment update or a passing test suite. Existing fits and their recorded environments remain intact. No study model fits are rerun for publication in this update.

## Validation

Frozen Python and Node installs, `uv pip check`, the offline lockfile check, Ruff, mypy across 537 source files, registry documentation checks, theme-integrity checks, Markdown formatting and spelling pass on Windows. The full test suite passes: 4,296 passed and 9 skipped. The dev-configuration ITT fit completes its release decision with zero divergences, maximum R-hat 1.006 and minimum ESS 577. These values show the sampler ran under the updated environment; the fit was not run for publication. Its Numba object-mode warning for beta-binomial draws also appears under PyTensor 3.3.7. A notebook converts through Jupytext with unchanged content and exports to HTML. Source and wheel builds pass, and archive inspection confirms that local environments, caches and outputs are excluded. No stored scientific results are regenerated.
