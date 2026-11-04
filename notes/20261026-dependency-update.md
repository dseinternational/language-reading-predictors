> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Dependency update, 26 October 2026

The shared research library advances from `v0.18.0` to `v0.19.2`, resolving to commit `1397003472170b5559df4f4652791c7db6a3957d`. The selected extras, Python 3.14 requirement and four supported platform environments are unchanged. The [shared upgrade notes](https://github.com/dseinternational/research/blob/v0.19.2/docs/migrating-to-0.19.md) describe the dependency minimums and figure palette changes. Several newer scientific minimums were already satisfied by this repository's previous lockfile, including Numba, NumPy, PyArrow, PyTensor, SHAP and DuckDB.

| Python package     | Previous version | Updated version |
| ------------------ | ---------------- | --------------- |
| alembic            | 1.20.0           | 1.20.1          |
| array-api-compat   | 1.15.0           | 1.16.0          |
| certifi            | 2026.10.19       | 2026.10.23      |
| dse-research-utils | 0.18.0           | 0.19.2          |
| jsonpointer        | 3.4              | 3.5             |
| pandas             | 3.0.6            | 3.1.0           |
| polars             | 2.0.0            | 2.1.0           |
| polars-runtime-32  | 2.0.0            | 2.1.0           |
| pycparser          | 3.11             | 3.12            |
| pyreadstat         | 1.3.6            | 1.3.7           |
| python-dateutil    | 2.9.0.post0      | 3.0.0           |
| ruff               | 0.17.1           | 0.17.2          |
| soupsieve          | 3.0.3            | 3.0.4           |
| sqlalchemy         | 2.1.5            | 2.1.6           |
| tornado            | 6.5.10           | 6.6             |
| wcwidth            | 0.9.3            | 0.9.4           |

The pnpm pin advances from 12.14.0 to 12.15.0 and the CI uv setup action from 10.3.0 to 10.3.2. CSpell 10.4.0, Prettier 3.9.10 and the other CI actions already use their latest stable releases. The Node lockfile also refreshes the C++, British English and US English dictionaries and the language-server text-document package. After resolution, the only installed Python package with a newer stable release is `cachetools`: PyMC requires it below version 7, so it remains at 6.2.6.

The shared library changes the categorical order from blue, green, orange to blue, orange, green. Explicit report roles now select the new positions to preserve the descriptions in the report: intervention and model predictions stay blue, control stays orange, and observed values and mechanism examples stay green. The prior/posterior overlay now specifies its blue/green palette explicitly. Regression tests check consistency across figures and the documented hues, so another palette reorder cannot silently swap these meanings. Default colour cycles elsewhere follow the updated shared style.

The [dateutil 3.0.0 release notes](https://github.com/dateutil/dateutil/releases/tag/3.0.0) describe changes to time-zone lookup and offsets for times in daylight-saving gaps. Searches found no direct dateutil use in the repository's Python source or scripts. A fixed-offset timestamp parsing check passed; this does not validate every time-zone behaviour in indirect consumers. [Polars 2.1.0](https://github.com/pola-rs/polars/releases/tag/py-2.1.0) rejects mixed string/numeric arithmetic. There are no direct Polars calls in the repository's Python source or scripts. [Pyreadstat 1.3.7](https://github.com/Roche/pyreadstat/releases/tag/v1.3.7) changes reported SPSS string widths; the source-provenance script reads values and discards that metadata.

Manual compatibility checks loaded the study data (216 rows and 146 columns), grouped all rows by child, read a pandas Parquet export through Polars and Arrow, preserved missing values, and round-tripped a small numeric table through SPSS. Representative synthetic arm-comparison, predictive-check and prior/posterior figures rendered successfully and were inspected visually. These checks do not establish compatibility with external databases or every SPSS schema. No scientific model fits were rerun, and stored fit artefacts were not regenerated.

Validation passed for `uv sync --locked`, `uv lock --check`, `uv pip check`, `pnpm install --frozen-lockfile`, `uv build`, Ruff over source and scripts, mypy over all 536 source files, the registry-documentation check, Markdown formatting and spelling. The full test suite passed: 4,285 tests passed and eight were skipped. The diagnostics and figure-colour tests also passed separately (87 tests); the final palette-argument adjustment passed its 10 focused tests and the manual rendering check.
