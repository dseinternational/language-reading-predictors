# Migrating the environment from conda to uv, 2026-08-22

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

**Scope.** Issue [#573](https://github.com/dseinternational/language-reading-predictors/issues/573): move this repository off the hybrid conda-forge + pip environment onto a single-layer [uv](https://docs.astral.sh/uv/) environment, following `dse-research-utils` v0.11.2 ([research#87](https://github.com/dseinternational/research/pull/87), and `docs/migrating-to-uv.md` in that repo). The conda layer existed only to supply a C toolchain and BLAS for PyTensor; [pymc#8318](https://github.com/pymc-devs/pymc/pull/8318) made Numba the default compile backend, so it no longer earns its cost, and every package in the former compiled core now ships a CPython 3.14 wheel.

This note records the five decisions a future reader might question, because several of them depart from — or go beyond — the instructions in the issue.

## 1. Writing `trace.nc` was unreachable from the issue's extras — fixed upstream, not here

The issue specified `boosting,columnar,dependence,graphs,io,notebook,tuning,viz`. That set did not install `h5netcdf`, and `uv tree --package h5netcdf --invert` confirmed the package reached this project **only** through `dse-research-utils[storage]`. Every statistical fit persists its posterior as `trace.nc` and re-reads it (`diagnostics.py`, `subfits.py`, `influence.py`, `sensitivity.py`, `release.py`, `blending_sensitivity.py`, and the `regenerate_*` scripts), so a repo taking those extras could sample a model and then fail to write the trace it had just produced. The old `environment.yml` carried `h5py` and `h5netcdf` unconditionally, so routing them through an extra was a change in reachability rather than a deliberate narrowing.

The first version of this branch worked around it by adding `storage`. That was the wrong layer, and the library agreed: [research#89](https://github.com/dseinternational/research/issues/89) moved `h5netcdf>=1.8.1` and `h5py>=3.16.0` into the base dependencies in **v0.11.1**, on the reasoning that netCDF is the on-disk format of the library's own output rather than an optional backend — `DataTree.to_netcdf`, which `InferenceData.to_netcdf` calls, accepts only the netcdf4 and h5netcdf engines and neither was otherwise reachable. `storage` now holds `zarr` alone, which nothing here imports, so this repo takes **exactly the extras the issue prescribed** and the workaround is gone. Verified after the bump: `h5netcdf` resolves through the base dependency, and an ArviZ `to_netcdf` / `from_netcdf` round-trip succeeds.

Worth recording because the unit suite does not catch this class of gap: the netCDF-touching tests either use small `az.from_dict` traces or stub `to_netcdf` outright. It was found by reading the import surface against the extras table, not by a failing test.

## 2. No `jax` extra — confirmed, not assumed

Every `pm.sample` call site in `src/` and `scripts/` passes `nuts_sampler="nutpie"` as a literal (34 occurrences); nothing reads a sampler name from run config, and nothing imports `jax`, `jaxlib` or `numpyro`. The old shared conda core installed the JAX stack everywhere regardless, so dropping it here removes an unused dependency rather than a capability. The one consequence is cosmetic: `repl_utils._REPORTED_PACKAGES` listed `numpyro`, `jax` and `jaxlib`, which would now print "Not found" in the notebook banner, so those three entries were removed.

This is the point of difference with `us-birth-certificates`, which keeps `jax` because its `nuts_sampler` is read from run config and numpyro can therefore be selected at runtime.

## 3. `environment-lock.json` moves to schema 2

`provenance.environment_lock()` recorded a `conda_packages` list read from `$sys.prefix/conda-meta/*.json`, plus the SHA-256 of `environment.yml`. Neither exists under uv. The record now pins the SHA-256 of `uv.lock` as `project_environment_spec` and drops `conda_packages` entirely; `python_distributions` is unchanged in form but is now the whole environment rather than only its pip layer, so no provenance is lost — the conda list and the distribution list always described the same packages from two sides. The schema version is bumped to 2 so a reader can tell a pre-migration publication from a post-migration one without inspecting the keys.

Existing published fits keep their schema-1 locks. Nothing reads the `conda_packages` key — `_technical.qmd` only names the file, and `release.py` does not open it — so no stored artefact is invalidated by the change and no refit is required.

## 4. The refit sweep's environment-identity check never matched (pre-existing bug, fixed here)

`scripts/run_refit_sweep.py` decides whether a stored fit can be reused on resume. Its docstring says the check covers "the environment lock digest", and `_reuse_reason` compared `config.json`'s `environment_lock_sha256` against `SweepIdentity.environment_sha256`. But that field was the digest of **`environment.yml`**, while the stored value is the digest of the serialised **`environment-lock.json`** — two different documents, so the comparison could never be true. Checked against a stored fit: `lrp-rli-adj-065-reporting` records `54abb909…` (its lock file), while `environment.yml` hashed to `73c072a3…`. Every statistical model therefore reported "environment.yml digest changed" and was refitted on every resume, silently making the driver's most expensive skip rule inert. That is consistent with what the August 2026 sweep record describes, and it would not have shown up as a failure — only as a sweep that redid work it had already done.

The migration had to touch that line anyway, so it is fixed rather than transliterated: `provenance` now exposes `environment_lock_payload()` and `environment_lock_sha256()`, `write_environment_lock` serialises through the former, and the sweep digests the same bytes the fit wrote. A test asserts the two agree, which is what makes the comparison correct by construction rather than by coincidence.

## 5. Adding Windows to CI exposed two latent portability bugs

Neither is caused by the migration; both were dormant because CI had only ever run on linux-64. They are fixed here because the Windows leg is added here.

**Text-mode line endings broke the environment lock's own integrity check.** `write_environment_lock` digested the in-memory `\n` payload but wrote it in text mode, which translates to `\r\n` on Windows — so the digest `config.json` records described bytes that were not the ones on disk, and `tests/statistical_models/test_itt_pipeline.py` (which re-hashes the file) failed. The payload is now written with `write_bytes`, so the digest is byte-exact by construction on every platform. On Linux the two were accidentally identical, which is why it never surfaced.

**Reads of repository text files used the locale encoding.** `path.read_text()` with no `encoding` decodes as cp1252 on Windows, and 3 of the 292 `docs/models/*/index.qmd` files contain a byte (0x9d) that cp1252 does not define — `lrp-rli-did-007`, `-011` and `-012`. The other 289 decoded by luck, not by design. Every read of a repository-tracked text file in `tests/` and `scripts/` now names `encoding="utf-8"`, as do the remaining unencoded text reads and writes in `src/`. Reads of JSON that `json.dump` produced are left alone: `ensure_ascii=True` makes those files pure ASCII, so they decode identically under any locale.

To keep this from regressing without a Windows machine to hand, the suite was run locally under `PYTHONUTF8=0 LC_ALL=C` — an ASCII default encoding, which is stricter than cp1252. That found two further cases the Windows job had not (the rendered-HTML reads in `test_report_restructure.py`, whose typography happens to be cp1252-representable). All 2283 tests now pass under it.

`.gitattributes` also pins `*.lock` to LF. Without it `* text=auto` plus the Windows runner's `core.autocrlf=true` would check `uv.lock` out with CRLF, giving the same committed file a different SHA-256 on Windows than on Linux — which would make the digest recorded in `project_environment_spec` platform-dependent and defeat the point of recording it.

## Consequences for contributors

- **Windows no longer needs WSL.** `jaxlib` was the package with no conda-forge win-64 build; native win-amd64 wheels now exist for the whole stack, and `windows-latest` is added to the CI test matrix.
- **Intel macOS is no longer supported.** Upstream's decision, not ours: numba publishes no macOS x86_64 wheels, and `shap` pins `numba<0.63` there. Apple Silicon, Linux and Windows are unaffected.
- **The system Graphviz `dot` binary is still needed** for model graphs. It is not a Python package: `brew install graphviz`, `apt install graphviz` or `winget install Graphviz.Graphviz`.
- **Grouped extras are coarser than the old per-package lists.** Taking `columnar` for `pyreadstat` also installs `polars` and `duckdb`, which this repo does not import. That is deliberate upstream — it keeps the Arrow-backed data layer in lockstep across repos.

## Verification

CI runs on `ubuntu-26.04` and `windows-latest`. `uv sync` resolves 194 packages against CPython 3.14.5 with `dse-research-utils` at tag v0.11.2; `ruff check src/ scripts/`, `python -m mypy` (36 files), `scripts/check_statistical_documentation.py` and the full `pytest` suite (2283 passed) all pass in the uv environment, as do `npm run spellcheck` and `npm run format:check`. No model was refitted for this change and none needs to be: nothing in the fitted numerics moved.
