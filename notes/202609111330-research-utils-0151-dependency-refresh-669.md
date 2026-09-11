> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Research utilities 0.15.1: a lock refresh with the compiled core held still

[Issue #669](https://github.com/dseinternational/language-reading-predictors/issues/669) moves the `dse-research-utils` pin from tag `v0.15.0` to `v0.15.1`, which resolves to commit `a16183955674720a011080b315ea1a97825f665a` — the merge commit of [upstream PR #105](https://github.com/dseinternational/research/pull/105), which the tag points at exactly. The release changes no library API and no dependency floor. It is a maintenance refresh of six locked third-party packages, and this repository's job is to re-resolve its own lock: Git consumers do not inherit the upstream one. The extras are unchanged. The upstream account is the [0.15.1 migration guide](https://github.com/dseinternational/research/blob/v0.15.1/docs/migrating-to-0.15.1.md).

## What moved

Seven packages, with nothing added or removed:

| Package              | Before              | After               |
| -------------------- | ------------------- | ------------------- |
| `dse-research-utils` | 0.15.0 (`e818caf5`) | 0.15.1 (`a1618395`) |
| `fonttools`          | 4.64.0              | 4.65.0              |
| `pure-eval`          | 0.2.3               | 0.2.4               |
| `ruff`               | 0.16.6              | 0.16.7              |
| `scikit-learn`       | 1.9.0               | 1.9.1               |
| `tqdm`               | 4.70.0              | 4.70.1              |
| `wrapt`              | 2.4.0               | 2.4.1               |

This is the exact set the issue predicted, at the exact versions it predicted, so the resolution needed no judgement beyond confirming it. The upgrade was targeted rather than a bare `uv lock --upgrade`, for the reason recorded under [#667](https://github.com/dseinternational/language-reading-predictors/issues/667): a dependency move should stay the move it claims to be and not silently absorb the weekly Dependabot sweep.

Upstream additionally locked `build` 1.6.1 and `uv` 0.12.13. Neither is present in this repository's lock and neither was added — they are repository-only tooling for the library's own release process, not something a consumer inherits.

## What did not move, and why that is the substance of this note

**NumPy 2.5.3, numba 0.67.0, PyTensor 3.3.1 and PyMC 6.3.2 are unchanged.** That is the compiled core every posterior is sampled through, and holding it still is what separates this refresh from #667, where lifting the NumPy ceiling moved all four and required a 276-model design-identity sweep to establish that no model had changed.

Nothing in the seven packages above participates in building a PyMC graph. `scikit-learn` is the only one used in modelling at all, and only in the gradient-boosting layer (`ml_utils.py`'s `RandomizedSearchCV` wrapper and the `GroupKFold` splitting); the other six are a font-metrics library behind Matplotlib, a REPL-safe expression evaluator behind IPython tracebacks, the linter, a progress bar and a decorator library. So the full identity sweep was **not** repeated here, and this note does not claim its result. The inference stands on the compiled core being byte-identical and no model-building code having changed in this branch, which is a narrower and weaker claim than #667's measured one — recorded plainly so that a future reader does not mistake the two.

## The environment this branch started from was already behind its own lock

`uv sync --locked` moved more than the lock diff did, which is worth recording because it looks alarming and is not. The seven packages above are what changed in `uv.lock`; the sync additionally installed PyMC 6.3.1 → 6.3.2, PyTensor 3.3.0 → 3.3.1, `pytensor-distributions` 0.2.0 → 0.3.2, `preliz` 0.27.1 → 0.28.0, `platformdirs` 4.11.5 → 4.11.7 and `ruff` 0.16.5 → 0.16.6 → 0.16.7.

Those are the #667 versions. The working environment had never been synced to the v0.15.0 lock that #668 merged, so it was carrying the pre-#667 compiled core while the lockfile declared the post-#667 one. The consequence is only that the "before" state on this machine was not the committed baseline; the "after" state is the committed lock exactly, and a second `uv sync --locked` reports no further changes. It does mean the validation below establishes that the suite passes under the 0.15.1 lock, not that it changed behaviour relative to the 0.15.0 lock as installed here, because the 0.15.0 lock was never installed here.

## Stored fits still refuse trace reuse

`environment_lock_sha256` digests the serialised environment snapshot — the installed distributions, not `uv.lock` — so it moves whenever any installed version moves, and it is a bound field of the reuse contract. Under this branch it is `693d56f718c917ebfee866724b005cf2c9b2471fd7ae4f4a29ef71e04057cea9`.

Every stored fit therefore continues to mismatch it and `--reuse-trace` remains refused, exactly as #667 recorded. This could not be checked against the artefacts themselves, because `output/` is absent from this checkout; the claim rests on the digest's definition rather than on a comparison. As under #667, the guard is conservative here rather than indicative: no model-building code changed, so the stored posteriors remain the posteriors of the models the code builds today. **Whether to schedule a rebuild is a decision for the maintainer, not a consequence of this PR.**

## Validation

- `uv lock --check` passes; `uv sync --locked` installed `dse-research-utils` 0.15.1, and the installed `direct_url.json` confirms both the requested tag `v0.15.1` and the commit `a1618395`.
- Imports confirmed for all eight declared extras — `boosting`, `columnar`, `dependence`, `graphs`, `io`, `notebook`, `tuning`, `viz` — and the core reports NumPy 2.5.3, PyMC 6.3.2, PyTensor 3.3.1, ArviZ 1.3.0, scikit-learn 1.9.1.
- `uv run pytest` — 3,453 passed, 4 skipped, 0 failed. The four skips are checkout and platform conditions, not version-sensitive: three need a populated `output/` (absent here) and one needs POSIX permission bits. #667 recorded one skip because that run had stored fits available.
- `uv run mypy` — no issues in 493 source files. `ruff check src/`, `npm run format:check` and `npm run spellcheck` pass.
- Upstream gating confirmed before starting: PR #105 merged, tag `v0.15.1` on that merge commit, and its Linux checks (`test-python (ubuntu-26.04-arm)`, `build-python`, `spellcheck`) all green. The upstream PR's four Windows failures are in the library's own suite and concern filename, symlink and concurrent-replacement cases; this repository's suite passes on Windows, which is where it was run.

No fits were run and no stored artefact was modified. Refits and publication remain separate work.

## An unrelated fix carried along

`npm run spellcheck` was already failing on `main` at [ac58282](https://github.com/dseinternational/language-reading-predictors/tree/ac5828228fb8dd1dcddf8d198de9fd097ea94baf) — `pkill` in `notes/202609080119-full-rebuild-both-layers.md`, from the rebuild note merged under #666. It is a real command name, so it went to `config/spellcheck/allow-en.txt` as the conventions direct rather than being reworded. Unrelated to this upgrade, and fixed here only because the documented pre-commit checks must pass and bypassing them is not an option.
