# Migrating the environment from conda to uv, 2026-08-22

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

**Scope.** Issue [#573](https://github.com/dseinternational/language-reading-predictors/issues/573): move this repository off the hybrid conda-forge + pip environment onto a single-layer [uv](https://docs.astral.sh/uv/) environment, following `dse-research-utils` v0.11.0 ([research#87](https://github.com/dseinternational/research/pull/87), and `docs/migrating-to-uv.md` in that repo). The conda layer existed only to supply a C toolchain and BLAS for PyTensor; [pymc#8318](https://github.com/pymc-devs/pymc/pull/8318) made Numba the default compile backend, so it no longer earns its cost, and every package in the former compiled core now ships a CPython 3.14 wheel.

This note records the four decisions a future reader might question, because three of them depart from — or go beyond — the instructions in the issue.

## 1. The `storage` extra is required, and the issue's extras list omitted it

The issue specified `boosting,columnar,dependence,graphs,io,notebook,tuning,viz`. That set does not install `h5netcdf`, and `uv tree --package h5netcdf --invert` confirms the package reaches this project **only** through `dse-research-utils[storage]`. Every statistical fit persists its posterior as `trace.nc` and re-reads it (`diagnostics.py`, `subfits.py`, `influence.py`, `sensitivity.py`, `release.py`, `blending_sensitivity.py`, and the `regenerate_*` scripts), so without a netCDF backend the first `to_netcdf` of a real fit would fail. The old `environment.yml` carried `h5py` and `h5netcdf` explicitly, so this is a transcription gap in the issue rather than a change of intent. The declared set is therefore `boosting,columnar,dependence,graphs,io,notebook,storage,tuning,viz`.

The unit suite does not catch this: the tests that touch netCDF either use small `az.from_dict` traces that happen to round-trip through whatever backend is present, or stub `to_netcdf` outright. It was found by reading the import surface, not by a failing test.

## 2. No `jax` extra — confirmed, not assumed

Every `pm.sample` call site in `src/` and `scripts/` passes `nuts_sampler="nutpie"` as a literal (34 occurrences); nothing reads a sampler name from run config, and nothing imports `jax`, `jaxlib` or `numpyro`. The old shared conda core installed the JAX stack everywhere regardless, so dropping it here removes an unused dependency rather than a capability. The one consequence is cosmetic: `repl_utils._REPORTED_PACKAGES` listed `numpyro`, `jax` and `jaxlib`, which would now print "Not found" in the notebook banner, so those three entries were removed.

This is the point of difference with `us-birth-certificates`, which keeps `jax` because its `nuts_sampler` is read from run config and numpyro can therefore be selected at runtime.

## 3. `environment-lock.json` moves to schema 2

`provenance.environment_lock()` recorded a `conda_packages` list read from `$sys.prefix/conda-meta/*.json`, plus the SHA-256 of `environment.yml`. Neither exists under uv. The record now pins the SHA-256 of `uv.lock` as `project_environment_spec` and drops `conda_packages` entirely; `python_distributions` is unchanged in form but is now the whole environment rather than only its pip layer, so no provenance is lost — the conda list and the distribution list always described the same packages from two sides. The schema version is bumped to 2 so a reader can tell a pre-migration publication from a post-migration one without inspecting the keys.

Existing published fits keep their schema-1 locks. Nothing reads the `conda_packages` key — `_technical.qmd` only names the file, and `release.py` does not open it — so no stored artefact is invalidated by the change and no refit is required.

## 4. The refit sweep's environment-identity check never matched (pre-existing bug, fixed here)

`scripts/run_refit_sweep.py` decides whether a stored fit can be reused on resume. Its docstring says the check covers "the environment lock digest", and `_reuse_reason` compared `config.json`'s `environment_lock_sha256` against `SweepIdentity.environment_sha256`. But that field was the digest of **`environment.yml`**, while the stored value is the digest of the serialised **`environment-lock.json`** — two different documents, so the comparison could never be true. Checked against a stored fit: `lrp-rli-adj-065-reporting` records `54abb909…` (its lock file), while `environment.yml` hashed to `73c072a3…`. Every statistical model therefore reported "environment.yml digest changed" and was refitted on every resume, silently making the driver's most expensive skip rule inert. That is consistent with what the August 2026 sweep record describes, and it would not have shown up as a failure — only as a sweep that redid work it had already done.

The migration had to touch that line anyway, so it is fixed rather than transliterated: `provenance` now exposes `environment_lock_payload()` and `environment_lock_sha256()`, `write_environment_lock` serialises through the former, and the sweep digests the same bytes the fit wrote. A test asserts the two agree, which is what makes the comparison correct by construction rather than by coincidence.

## Consequences for contributors

- **Windows no longer needs WSL.** `jaxlib` was the package with no conda-forge win-64 build; native win-amd64 wheels now exist for the whole stack, and `windows-latest` is added to the CI test matrix.
- **Intel macOS is no longer supported.** Upstream's decision, not ours: numba publishes no macOS x86_64 wheels, and `shap` pins `numba<0.63` there. Apple Silicon, Linux and Windows are unaffected.
- **The system Graphviz `dot` binary is still needed** for model graphs. It is not a Python package: `brew install graphviz`, `apt install graphviz` or `winget install Graphviz.Graphviz`.
- **Grouped extras are coarser than the old per-package lists.** Taking `columnar` for `pyreadstat` also installs `polars` and `duckdb`, which this repo does not import. That is deliberate upstream — it keeps the Arrow-backed data layer in lockstep across repos.

## Verification

`uv sync` resolves 194 packages against CPython 3.14.5; `ruff check src/ scripts/`, `python -m mypy` (36 files), `scripts/check_statistical_documentation.py` and the full `pytest` suite (2283 passed) all pass in the uv environment, as do `npm run spellcheck` and `npm run format:check`. No model was refitted for this change and none needs to be: nothing in the fitted numerics moved.
