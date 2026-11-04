> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Dependency update, 28 October 2026

The Python lockfile advances 27 packages, including `dse-research-utils` from 0.18.0 to 0.19.3 at commit `f881f5830fc68896a888651a08b78898111ba04a`. The selected extras and supported Python 3.14 platforms are unchanged. Build tooling advances to Hatchling 1.33.0, the pytest minimum to 9.2.0 and the Ruff minimum to 0.17.3. Scientific dependency requirements remain owned by the shared library.

| Package            | Previous version | Updated version |
| ------------------ | ---------------- | --------------- |
| alembic            | 1.20.0           | 1.20.1          |
| array-api-compat   | 1.15.0           | 1.16.0          |
| azure-identity     | 1.26.0           | 1.27.0          |
| certifi            | 2026.10.19       | 2026.10.23      |
| dse-research-utils | 0.18.0           | 0.19.3          |
| fonttools          | 4.67.0           | 4.68.0          |
| h5netcdf           | 1.8.1            | 1.8.2           |
| jsonpointer        | 3.4              | 3.6             |
| jupyter-builder    | 1.2.3            | 1.2.4           |
| narwhals           | 2.27.1           | 2.28.0          |
| obstore            | 0.12.0           | 0.12.1          |
| pandas             | 3.0.6            | 3.1.0           |
| platformdirs       | 4.13.5           | 4.13.7          |
| polars             | 2.0.0            | 2.1.0           |
| polars-runtime-32  | 2.0.0            | 2.1.0           |
| pycparser          | 3.11             | 3.12            |
| pyreadstat         | 1.3.6            | 1.3.7           |
| pytest             | 9.1.1            | 9.2.0           |
| python-dateutil    | 2.9.0.post0      | 3.0.0           |
| pywinpty           | 3.1.0            | 3.1.1           |
| ruff               | 0.17.1           | 0.17.3          |
| soupsieve          | 3.0.3            | 3.0.4           |
| sqlalchemy         | 2.1.5            | 2.1.6           |
| tornado            | 6.5.10           | 6.6             |
| tzdata             | 2026.5           | 2026.6          |
| wcwidth            | 0.9.3            | 0.9.4           |
| zarr               | 3.4.1            | 3.4.2           |

The pnpm pin advances from 12.14.0 to 12.15.0 and its lockfile refreshes the C++, British English and US English dictionaries and the language-server text-document dependency. CSpell 10.4.0 and Prettier 3.9.10 remain the latest stable direct Node dependencies. The uv setup action advances from 10.3.0 to 10.3.2; checkout, Node setup and pnpm setup already use their latest stable releases.

The resolver retains Numba 0.68.0 because PyTensor requires `numba<=0.68.0`, and Cachetools 6.2.6 because PyMC requires `cachetools<7`. These bounds prevent the available Numba 0.68.1 and Cachetools 7.2.6 upgrades. No resolver overrides weaken the scientific stack's compatibility requirements.

## Compatibility review

The shared library's [0.19.3 migration guide](https://github.com/dseinternational/research/blob/v0.19.3/docs/migrating-to-0.19.md) covers the pandas 3.1 and Polars 2.1 changes, the reordered categorical palette and earlier dependency changes since 0.18.0. DuckDB 2.0 was already installed by the preceding repository update. Searches found no direct DuckDB, Polars or dateutil calls in this repository's source modules; the SPSS source-audit script uses pyreadstat.

An isolated pandas 3.0.6 environment reproduced the original loaded-data CSV digest. Comparing its loaded frame with pandas 3.1.0 using `pandas.testing.assert_frame_equal(check_exact=True)` found the same 216 rows and 146 columns, cell values, data types, categories and missing values. CSV bytes do differ because pandas 3.1 prints some floating-point values with different decimal representations. A CSV digest alone would therefore incorrectly suggest changed loaded values. The source data bytes and historical fit manifests remain intact; this check does not certify saved-fit reuse or scientific results under the updated environment.

The library now orders categorical colours blue, orange, green, purple, teal and pink. Role assignments move the wait-list control to `CHART_COLOURS[1]` and observations, worked examples and readiness knees to `CHART_COLOURS[2]`, preserving their orange and green hues. Intervention and model summaries remain blue. Regression checks now test hue ranges as well as consistent palette positions, so another palette reorder cannot silently change those meanings. Representative arm-overlap figures were rendered from synthetic draws and inspected.

The copied Quarto theme and pin advance together from `tokens/2026.3.4` to `tokens/2026.3.7`. All eleven theme, font and licence hashes match the upstream release manifest; the font bytes are unchanged. A temporary Quarto HTML document rendered with both theme layers and native MathML. The local uv cache is explicitly ignored at the repository root because the package-build check found that Hatchling otherwise included it in the source archive despite the cache's own ignore file.

## Validation

Both frozen install checks, `uv pip check`, the offline Python lock check, Ruff, whole-package mypy (537 source files), registry documentation checks, theme-integrity checks, Markdown formatting and spelling passed. All 192 installed pinned Python versions and the shared library's source tag and commit match the lock. The nine figure-colour regression tests passed. Source and wheel builds passed, and archive inspection confirmed that the local uv cache is excluded. See the PR checks for the complete test-suite result. No scientific model fits were rerun.
