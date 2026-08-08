> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# The family split begins: a shared artefact layer, and ITT + joint into `pipelines/` (#394, tranche 4)

**Date:** 2026-08-07. **Issue:** #394 (complete the statistical pipeline family split and artefact lifecycle refactor), implementation-sequence step 5 plus the part of design point 6 it depends on. **Change:** pure relocation — no estimand, likelihood, prior, analysis population, fitted equation, sampling preset, diagnostic threshold or artefact schema is touched.

## What this tranche does

Tranches 1–3 ([202608072130](202608072130-394-artifact-interface.md)) gave the pipeline one artefact interface, a fit-level manifest and a single expression of the primary-fit lifecycle. This one starts moving families out of the monolith. `fit_itt` (with its 393-line floor branch) and `fit_joint` now live in `statistical_models/pipelines/itt.py` and `pipelines/joint.py`; `pipeline.py` re-exports both, so every model module and test keeps its import path. `pipeline.py` drops from 10,182 lines to 7,478.

## Why the shared layer had to move first

The obvious version of step 5 — cut `fit_itt` out and import what it needs from `pipeline.py` — does not work: `pipeline.py` must re-export `fit_itt`, so `pipelines/itt.py` importing `pipeline` closes an import cycle. The question is therefore not "can ITT move" but "what does ITT reach, and who else reaches it".

An AST reachability scan over the 25 `fit_*` entry points answered it. `fit_itt` and `fit_joint` transitively reach 42 module-level functions. Ten are theirs alone (996 lines, the two entry points and the floor branch included). The other 32 — 1,175 lines — are reached by 24 other family entry points as well: the posterior-predictive suite, the prior panel and pushforwards, the ROPE/forest/trajectory figure wrappers, the console banners and report-template copy, and the thin per-phase stage wrappers. Those had to become a layer _below_ the families, not a peer of them. That is design point 6 of the issue ("move shared prior artefacts, PPC artefacts, treatment-effect/ROPE figures, trajectories and report-template publication into cohesive modules named by responsibility"), and it is a prerequisite for every subsequent family move, not just this one.

The scan also confirmed the partition is clean: **no function in the moved set references anything that stays in `pipeline.py`**. The dependency graph across the new modules is acyclic and one-directional — `publication`, `prior_artifacts`, `ppc_artifacts`, `figure_artifacts` and the influence helpers have no new-module dependencies; `runtime` depends on those four plus `diagnostics`; `pipelines/itt` depends on `runtime`, `publication` and `figure_artifacts`; `pipelines/joint` adds `prior_artifacts` and `pipelines/itt`.

## The new modules

| Module                | Holds                                                                                                                                                         | Lines |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| `publication.py`      | start/end-of-fit banners, the LOO summary row, the `index.qmd` + `_partials/` copy, the Graphviz model-graph render                                           | 143   |
| `prior_artifacts.py`  | the pruned prior panel and `priors_table.csv` (with the per-kind constructor/role/rationale overrides), and the estimand-scale prior-pushforward row builders | 809   |
| `ppc_artifacts.py`    | the #318 posterior-predictive suite: coverage CSV, calibration panel, count/off-floor/legacy overlay routing                                                  | 364   |
| `figure_artifacts.py` | the fit-context side of the report figures — ROPE, forests, predicted scores, arm overlap, trajectories, child fits, the contrast heatmap                     | 511   |
| `runtime.py`          | `shared_stages()` (the `StageHooks` binding), the per-phase wrappers every family calls, and `require_spec`                                                   | 121   |
| `pipelines/itt.py`    | `fit_itt`, `fit_itt_floor_rule`, the ITT diagnostic contract, the family's audit bindings and its Area 1/4 extras                                             | 762   |
| `pipelines/joint.py`  | `fit_joint`                                                                                                                                                   | 280   |

The influence helpers (`influence_diagnostics`, `write_loo_influence`) went into the existing `diagnostics.py` rather than a new module: they read Pareto-k off a fitted trace and persist `pareto_k.csv`, which is what that module is for. Note that the pre-existing `influence.py` is a different thing — the post-hoc refit tool for influential children.

`figure_artifacts.py` is deliberately the _fit-context_ side only. The drawing already lived in focused modules (`effect_plots.py`, `predicted_scores.py`, `arm_overlap.py`, `trajectory_plots.py`) which the moved wrappers reach through function-local imports; those imports moved verbatim and the drawing modules are untouched.

### `runtime.py`, and why the binding is not in `stages.py`

`stages.py` defines the invariant lifecycle without knowing how any artefact is produced — that inversion is what `StageHooks` exists for, and it is what lets the lifecycle-ordering tests run without importing a single artefact writer. Binding the hooks inside `stages.py` would undo it. `runtime.py` is where the lifecycle and the concrete producers meet, and nothing else.

## Public names

Thirty-nine cross-module interfaces lost their leading underscore (`_save_rope_plot` → `save_rope_plot`, `_run_ppc` → `run_ppc`, and so on) — the issue asks for exactly this ("promote any cross-module private interfaces used by pipelines to supported public functions before moving their callers"). Names used only inside their new module stayed private (`_prior_table_overrides`, `_graphviz`, `_ppc_overlay_figure`, `_ctx_pareto_k`, …).

Two ITT wrappers were renamed rather than promoted: `_write_itt_analysis_audit` / `_write_itt_ppc_calibration` became `write_analysis_audit` / `write_ppc_calibration` in `pipelines/itt.py`, because the straightforward public names are already taken by the family implementations they wrap in `statistical_models/itt.py`. The distinction is real and worth the two names: `itt.write_itt_analysis_audit` is the audit; `pipelines.itt.write_analysis_audit` is the pipeline-level binding that supplies the loader.

## How the relocation was made checkable

Moving 2,685 lines by hand invites exactly the silent error this issue warns about ("equation or output changes hidden inside large file-movement diffs"). So the move was scripted, and the script asserts:

- **Region tripwires.** Each of the 19 cut regions is declared as a line range with its expected opening line, and must end exactly on a top-level statement boundary (checked against the AST) with a blank line after it. A range that bisected a function, or swallowed a neighbour, fails before anything is written.
- **No leakage.** After the cut, no moved function may still be defined in `pipeline.py`.
- **Byte-identity.** Re-applying the rename map to the original line ranges reproduces the new files exactly: 15 of the 17 moved regions match byte-for-byte, and the two that differ do so only in five docstring lines of `runtime.py`, refreshed deliberately afterwards because "compatibility wrapper for the shared attach stage" had stopped being true of a module whose whole purpose is that binding.
- **No string was renamed.** Tokenising the pre-move `pipeline.py` and searching only `STRING` tokens for the 40 old names returns exactly one hit — a `:func:` docstring cross-reference, which should update. No artefact filename, table key, console label or guard message was touched by the rename.

Ruff (`src/ scripts/ tests/`) is the backstop for the import wiring: every name the moved code needs and every import the monolith no longer needs surfaced as F821/F401 rather than being guessed at.

## Tests

`tests/statistical_models/test_pipeline_boundaries.py` locks the invariant that makes the rest of the split possible: neither the shared layer nor anything under `pipelines/` may import `pipeline.py` (checked by AST over both module-level and function-local imports), `pipeline.fit_itt is pipelines.itt.fit_itt`, and the migrated entry points are no longer defined in the monolith. A back-edge would silently re-tangle the package without failing anything else in the suite.

The ITT pipeline tests needed real updating, not just renaming: `monkeypatch.setattr(pipeline, "_run_ppc", …)` patches a module attribute, and `fit_itt` now resolves that name in its own module namespace. The `fast_pipeline` fixture now installs each stub on the module that actually resolves the name — which is itself informative, because it exposes that `emit_priors` is reached through the stage binding (`runtime`) rather than through the family module. The lifecycle-ordering test keeps its exact 13-event expectation.

## Verification

Six dev fits spanning the touched surfaces — `lrp-rli-itt-001` (ordinary ITT), `lrp-rli-itt-009` (the floor branch), `lrp-rli-itt-012` (joint), plus `lrp-rli-gf-001`, `lrp-rli-did-001` and `lrp-rli-mech-056` as untouched families that exercise the relocated shared layer — were run from `main` in a detached worktree and again from this branch. Every CSV is byte-identical. Full suite, `ruff check src/ scripts/ tests/`, `npm run format:check` and `npm run spellcheck` pass.

As established in tranche 1, SVG figures, `model_graph.png`, `trace.nc` and `config.json` are not byte-stable run to run (matplotlib ids and dates, Graphviz layout, timestamps); the CSVs are, which is why they are the comparison surface.

## What is not in this tranche

The other 23 families stay in `pipeline.py` for now, and with them the specialised algorithms design point 7 wants isolated (the longitudinal correlated-factor likelihood and log-prior recovery). `pipeline.py` is not yet the "small documented compatibility facade" of the first acceptance criterion — it is a smaller monolith with two families lifted out and a documented facade role for those two. The `SubfitRunner` (design point 5), the release-decision boundary (point 3), typed settings for the remaining families (point 4) and the MyPy gate (step 7) are all still open.

The natural next tranche is the first family _group_ — DiD, gain/level factors, aligned and block exposure (step 6) — which now has a landing surface: they adopt `PrimaryFitPlan` as they move, and the shared layer they need already exists.
