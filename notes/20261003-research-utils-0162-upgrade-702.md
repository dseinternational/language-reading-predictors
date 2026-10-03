> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Research utilities 0.16.2 dependency upgrade

[Issue #702](https://github.com/dseinternational/language-reading-predictors/issues/702) upgrades `dse-research-utils` from `v0.16.1` to `v0.16.2`. [Research PR #116](https://github.com/dseinternational/research/pull/116) is merged, its checks passed, and the published annotated tag resolves to its merge commit, `0b51bcc5372bfe0d57297b9806abedf4c7fa9375`. The tag object is `7d266f416a58cef0e5aafab80819a4ec26cb965e`. The [upstream upgrade notes](https://github.com/dseinternational/research/blob/v0.16.2/docs/migrating-to-0.16.2.md) describe the new minimum dependency versions and confirm that the library API and Python 3.14 requirement are unchanged.

## Changes

The only change to `pyproject.toml` is the research source tag. The eight selected extras and all other project constraints are unchanged. `uv lock --upgrade-package dse-research-utils` changed four packages, with none added or removed.

| Package            | Previous lock | Updated lock |
| ------------------ | ------------- | ------------ |
| dse-research-utils | 0.16.1        | 0.16.2       |
| azure-identity     | 1.25.3        | 1.26.0       |
| azure-storage-blob | 12.30.3       | 12.31.0      |
| pytensor           | 3.3.2         | 3.3.3        |

Xarray 2026.9.0 and DuckDB 1.5.6 already met the release's new minimums in the previous lock. The targeted refresh keeps the other locked versions. It does not copy the research repository's lockfile or update this project's development tools to the upstream versions. NumPy retains its `<2.6` ceiling, PyTensor retains `<3.4`, and Numba remains at 0.67.0.

## Installed environment

`uv lock --check` and `uv sync --locked` passed. The installed distribution and `dse_research_utils.__version__` both report 0.16.2. The installed `direct_url.json` records the requested tag and the release commit above. The local environment had fallen behind the previous committed lock, so installation changed more packages than the four-package lock diff. Validation uses the updated locked environment.

The checks ran on Apple silicon macOS with Python 3.14.8. The installed modelling and output packages report the following versions.

| Package    | Version  |
| ---------- | -------- |
| numpy      | 2.5.3    |
| numba      | 0.67.0   |
| llvmlite   | 0.49.0   |
| pymc       | 6.3.2    |
| nutpie     | 0.16.11  |
| arviz      | 1.3.0    |
| scipy      | 1.18.1   |
| xarray     | 2026.9.0 |
| duckdb     | 1.5.6    |
| h5netcdf   | 1.8.1    |
| h5py       | 3.16.0   |
| matplotlib | 3.11.2   |

## Validation

The full suite and the render-test rerun together passed all 4,215 runnable tests. One test skipped because the pre-#594 level-factor backup is absent from this checkout. The first complete run passed 4,206 tests; nine Quarto render tests failed because the sandbox blocked Jupyter from opening local kernel sockets. All nine passed when rerun with that permission. The suite covers model construction and log-probability compilation, prior and posterior prediction, plotting, temporary NetCDF writes, and simulated Azure uploads.

The registry documentation check, Ruff checks for `src/` and `scripts/`, Markdown formatting and spelling checks passed. Mypy reported no issues in 536 source files. `uv pip check` confirmed that all 194 installed packages have compatible requirements. The isolated fixed-seed ITT sampling test passed with the local compiler settings below. It compiles and runs the real PyMC/nutpie path, checks finite posterior values, and checks posterior and predictive array dimensions and score bounds. The small sampling run checks execution and shape compatibility, not convergence of a research fit.

Local tests use writable temporary cache directories and `PYTENSOR_FLAGS=base_compiledir=/private/tmp/lrp-702-pytensor-base,compiledir=/private/tmp/lrp-702-pytensor,cxx=`. PyTensor's optional C compiler is disabled because the default macOS command selects `-ld64`, which this host cannot link. The first isolated sampling test failed with `ld: library 'd64' not found`, as the [upstream release validation](https://github.com/dseinternational/research/pull/116) also reported. The test replaces `PYTENSOR_FLAGS` in its subprocess, so the rerun also sets `PYTENSORRC=/private/tmp/lrp-702-pytensorrc`, a temporary configuration file with an empty `cxx` setting under `[global]`. The production nutpie/Numba sampling backend remains active. The optional PyTensor C backend is not covered by this local validation. No repository compiler defaults or tests were changed.

## Scope

No research fits or published outputs were regenerated. The tests exercise synthetic models and temporary artefacts. Passing them checks software compatibility; it does not establish that a future refit will reproduce earlier numerical results. Existing fitted results retain their recorded environments. Refitting and publication remain separate work.
