> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Research utilities 0.13.0 upgrade and pending HSGP refits

The code migration for [issue #660](https://github.com/dseinternational/language-reading-predictors/issues/660) selects `dse-research-utils` tag `v0.13.0`, which resolves to commit `458cc41b1dc33f4c0204919253ac92251c61b2bb`. The lockfile retains the existing extras and inherits the shared dependency floors from the library. `uv sync --locked` installed that commit. The contract changes are documented in the [tagged migration guide](https://github.com/dseinternational/research/blob/v0.13.0/docs/migrating-to-0.13.md) and [upstream source review](https://github.com/dseinternational/research/blob/v0.13.0/docs/source-review-2026-09-06.md).

## Preserving the model during a refit

The Hilbert-space Gaussian process (HSGP) represents a smooth curve as a weighted sum of basis functions. The meaning of each weight depends on the basis size, domain half-width and domain midpoint. Dropping an extreme observation changes the subset's range and midpoint. Recomputing either quantity would therefore change the curve represented by the same weights.

The mechanism factory now calculates the midpoint and half-width from the full standardised inputs. It records the realised `m`, `L` and `center`, together with the exposure and moderator standardisers, in `extra.mechanism_design` in `config.json`. Subset construction passes these quantities directly to the shared constructor. The model's existing lengthscale and amplitude priors remain in place. The exact-LOO reader requires this stored design before rebuilding or evaluating a trace. A legacy record with only a boundary, or no record, requires a fresh full fit. The reader does not infer missing quantities from a subset. Age and phase-specific GP refits remain unsupported and fail explicitly; no registered affected model uses those paths.

The current wrapper records `hsgp_basis_version` in run metadata. Publication and comparison readers withhold HSGP results without the current marker. Both mechanism report-regeneration commands also refuse these old fits before reconstruction. The existing trace-reuse contract also compares the environment lock and model graph, so a new environment cannot silently reuse an old posterior.

## Other consumer changes

- The convergence reader distinguishes a failed or empty scan from a completed scan with unassessable parameter diagnostics. Both withhold findings. Null extrema cannot pass a threshold. Legacy summaries still require usable raw measurements.
- The upload adapter returns the complete `BlobUploadResult`. The upload command selects `report_url`, which identifies the root report. It retains encoded URLs verbatim. A repository search found no local filename comparison that strips upload URL prefixes; simulated uploads verify `relative_paths` against raw filenames containing spaces, plus signs, percent signs and non-ASCII characters.
- LOO influence counts now include every non-finite Pareto-k value once. A zero threshold remains zero, and non-finite thresholds are refused. Comparison checks no longer let a non-finite value disappear beside a finite maximum. Unassessable parameter diagnostics also fail an exact-refit convergence check.
- This repository computes relative efficiency through ArviZ rather than a local wrapper around the changed shared fallback. The changed shared `summarise_bands` helper has no direct consumer here; the existing interval imports retain their interfaces. Figure output already delegates to the shared writer. A regression test verifies that saving one collection leaves unrelated figures open. The numeric inverse-logit re-export has no symbolic model call sites.

## Refit inventory and status

[The inventory](assets/20260906-hsgp-refit-inventory-660.csv) lists all 20 registered models that currently use HSGP curves. All belong to the mechanism family. Each has a local reporting fit. One additional saved directory is an archived pre-refit copy of MECH-204, not another registered model. No registered ITT or joint model currently enables an HSGP term.

The inventory was constructed from repository base commit `6b4bc15fd387ea3c3ff121e7fca5039b7a2c2f48` with the #660 migration applied. It records the SHA-256 of each old configuration and the trace hash stored in that configuration. The recorded trace hashes were not independently recomputed. The new basis values and analysis-row counts come from constructing each current model against the local data under the installed release. This construction check does not sample a posterior or evaluate old weights.

**All 20 reporting refits and their held-out comparisons remain pending under [#660](https://github.com/dseinternational/language-reading-predictors/issues/660).** The implementation checks below do not qualify those scientific results. Do not close the issue or accept results under the new approximation until these fits have been sampled, checked and compared. Existing saved artefacts have been preserved. The publication and comparison readers mark their HSGP results as pending when read by the updated code; already exported static reports require regeneration as part of the refit work.

For each inventory row, run `uv run python scripts/fit_statistical_model.py <model_id> --config reporting` with the new lockfile, without trace reuse. Then rebuild the applicable mechanism comparisons with `scripts/compare_statistical_models.py`. Any exact-LOO repair must use the newly saved design and clear the refit convergence check. Record the new trace identities and comparison outcomes before changing the inventory status.

## Validation

The migration tests execute the installed shared constructor with fixed nonzero weights on asymmetric inputs. They cover removal of either extreme, a one-row subset, saved-design round trips, factory standardisation and prior preservation, and rejection of legacy designs before any trace evaluation. Diagnostic tests cover completed-but-unassessable, failed, empty, null and legacy summaries. Upload tests simulate Azure clients and cover root and nested reports and encoded filenames.

The initial test attempts encountered a read-only compiler cache and the macOS graphical plotting backend. Quarto also required local Jupyter socket connections that the sandbox denied. The completed suite ran outside that restriction with writable temporary caches and `MPLBACKEND=Agg`.

- `uv sync --locked` installed version 0.13.0 at the exact released commit. Installed package metadata confirms both the tag and commit.
- `uv run --locked pytest` passed 3,394 tests and skipped one test whose archived pre-#594 LF-006 fit is absent. Quarto rendering tests passed.
- Follow-up checks passed after the final changes. These included 172 refit, publication and regeneration tests; the added LOO non-finite-count tests and the retained low-BFMI NumPy-array regression; and the real held-out-density test. The held-out density agrees with an independent Beta-Binomial calculation.
- Ruff, mypy, Markdown formatting, spelling and `git diff --check` passed. Mypy checked 485 source files.

Fresh reporting fits and production held-out comparisons are not part of these implementation tests and remain pending above.
