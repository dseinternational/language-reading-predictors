> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Dependency review, 24 September 2026

Remove `.github/dependabot.yml` and refresh the repository's Python, npm and GitHub Actions dependencies. This removes the scheduled version-update configuration. It does not change GitHub's separate security-alert settings.

## Updates

| Dependency                   | Previous   | Updated    |
| ---------------------------- | ---------- | ---------- |
| Hatchling build requirement  | `>=1.32.0` | `>=1.32.4` |
| mypy development requirement | `>=1.18.2` | `>=2.3.1`  |
| Ruff development requirement | `>=0.16.1` | `>=0.16.8` |
| CSpell                       | `10.2.2`   | `10.3.3`   |
| Prettier                     | `3.9.6`    | `3.9.9`    |
| `astral-sh/setup-uv`         | `v10.0.1`  | `v10.2.0`  |

Regenerating `uv.lock` with `uv lock --upgrade` updates 37 packages and adds `msgspec`, required by Zarr 3.4. The mypy version was already locked at 2.3.1; its declared minimum now matches that version. The npm lock updates 42 package entries and removes the unused `fast-json-stable-stringify` entry. Package sources and supported Python platforms remain unchanged.

The review checked all 196 registry packages in the Python lock against the [PyPI JSON API](https://docs.pypi.org/api/json/), including packages selected only on other operating systems. It checked npm's direct and transitive dependencies with `npm outdated --all`, and all three GitHub Actions against their upstream releases. The shared library remains at its latest tag, [`dse-research-utils` v0.15.2](https://github.com/dseinternational/research/releases). [`actions/checkout` v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1) and [`actions/setup-node` v7.0.0](https://github.com/actions/setup-node/releases/tag/v7.0.0) are already current.

The build uses Python 3.14 and the npm tools use Node 24, as declared by the repository. Their existing version ranges admit maintenance releases. System tools such as Graphviz and Quarto are not pinned or installed by a repository dependency manifest.

## Updates held by upstream requirements

| Package        | Retained | Available | Reason                                                                                              |
| -------------- | -------- | --------- | --------------------------------------------------------------------------------------------------- |
| `cachetools`   | `6.2.6`  | `7.2.0`   | PyMC 6.3.2 requires `cachetools>=4.2.1,<7`.                                                         |
| Nested `chalk` | `5.6.2`  | `6.0.0`   | `chalk-template` 1.1.2 requires `chalk^5.2.0`. CSpell's direct Chalk dependency already uses 6.0.0. |
| `ini`          | `6.0.0`  | `7.0.0`   | `global-directory` 5.0.0 requires exactly 6.0.0.                                                    |

Keep these requirements intact. Forcing newer versions would override the dependency contracts declared by their maintainers. The Python requirements are available in [PyMC's release metadata](https://pypi.org/pypi/pymc/6.3.2/json). The npm requirements are in the published manifests for [`chalk-template`](https://registry.npmjs.org/chalk-template/1.1.2) and [`global-directory`](https://registry.npmjs.org/global-directory/5.0.0).

Review of the [Filelock 4.0 release](https://github.com/tox-dev/filelock/releases/tag/4.0.0) found a change to soft read/write locking. The repository has no direct Filelock use. [Zarr 3.4](https://github.com/zarr-developers/zarr-python/releases/tag/v3.4.0) adds metadata validation through `msgspec`. The [setup-uv release](https://github.com/astral-sh/setup-uv/releases/tag/v10.2.0) includes current uv checksums and a change to cache saving for merge queues.

## Validation

Validation uses Windows, Python 3.14.7 and Node 24.19.0. Temporary files and compiler caches are kept in this workspace because the first test attempt could not access the shared Windows temporary directory. That interrupted attempt is not a passing test result.

The completed checks passed:

- `uv sync --locked` and `uv pip check`.
- Full `uv run pytest --basetemp=tmp/pytest-dependency-update` run, with 3,864 passed, 4 skipped and 499 warnings in 14 minutes 19 seconds. No source files changed during this run.
- `uv run ruff check src/ scripts/`.
- `uv run python -m mypy`, covering 536 source files.
- `uv run python scripts/check_statistical_documentation.py`.
- `uv build`, producing a source distribution and wheel. The source distribution was checked for accidental cache inclusion.
- `npm ci`, `npm run format:check` and `npm run spellcheck`.
- `npm audit`, with no known vulnerabilities reported.
- `git diff --check`.

No production models, predictor rankings or scientific estimates were regenerated. Resolving the lock for the declared platforms does not test execution on those other platforms.
