# Joint-mechanism reporting refits after the code-review fix batch, 2026-08-21

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

**Run.** `lrp-rli-jm-001` and `lrp-rli-jm-002` refitted sequentially at `--config reporting` (6 chains × 6000 draws, `target_accept` 0.95) with `--render`, on branch `jm-family-code-review-fixes` at `c3c5fda8` (PR #568 — the joint-mechanism review fix batch, whose findings 1–3 changed the fitted models: jm-001's `hs_missing` covariate and Normal(0, 1) group-nuisance prior, the levels design's PSIS-LOO removal, and jm-002's cell-to-child LOO map). Data `rli_data_long.csv` sha `dc8dda5780b7…` (unchanged). Both fits exited 0 and rendered. The pre-refit reporting directories are preserved at `output/statistical_models/models_backup_20260821/` (with traces), per the back-up-before-remediation rule. 2 fitted, 0 failed; 2/2 convergence passes; 2/2 publishable.

## Gates and release

- **jm-001**: convergence gate passed (r̂, ESS, divergences, BFMI all clean); all four wave fits converged with zero divergences (max r̂ 1.0010, min ESS 5555, n = 53/53/53/52 at t1–t4; t3 anchor). `release_decision.json` publishable, settled at the robustness stage.
- **jm-002**: convergence gate passed; publishable at the robustness stage.

## The fix batch is live in the artefacts

- jm-001's `config.json` records `active_adjustment: [blocks, hs, hs_missing]` and `compute_loo: false`; the standard `effective_adjustment` record lists G, A, blocks, hs, hs_missing as fitted; **no** `loo.txt` / `pareto_k` / LOO-PIT artefacts exist (by design — the saturated levels residual; the recipe and results partial say why) while `psense_summary.csv` is present via the direct log-density route; `priors_table.csv` shows `beta_group_nuisance` as Normal(0, 1) (now also the fitted prior) with role nuisance, and `u_resid_chol` / `u_resid_z` as documented nuisances.
- jm-002's LOO is now genuinely leave-one-child-out: `pareto_k.csv` has 53 child rows (was 153 child-by-transition rows), 2 flagged above 0.7 (max k 0.94 — the usual conditional-on-`u_child` caveat for two influential children, not the mass failure of the levels design), child-unit `elpd_loo` −590.5. `effective_adjustment` lists G, A, hs, hs_missing, attend, deapp_c, deapp_c_missing plus both autoregressive baselines `W_pre` / `N_pre`. `priors_table.csv` shows `beta_G` role **association** (was `causal`) and the `u_child_*` block documented. The added `pre_required=("W", "N")` left the fitted rows identical (n_obs 153, 53 children — the unused `L_pre` requirement bound on zero rows, as the review predicted). `loo_pit.png` / `loo_pit_n.png` are written and the partial now displays the nonword sibling.

## Results — stable, as expected

**jm-001** (levels; adding `hs_missing` and widening the group nuisance): every published median moved by ≤ 0.04 and no interval or direction reading changed. Headlines at the clearest wave (t1): Δ = β(LS→N) − β(LS→W) = **−0.47** (50% −0.63 to −0.30; 89% −0.97 to +0.04), P(Δ > 0) = 0.07 — moderate evidence the _levels_ association favours word reading, the documented levels-scale reading (shared reading-development component + the 6-item nonword floor), not a contradiction of the ANCOVA contrast. Share retained: t1 0.90, t2 0.76, t3 0.82, t4 0.60 (medians). Within-wave residual correlation at t1 +0.40 (89% +0.07 to +0.67). The refit's new-child marginal coverage is 73% at the 50% level and 98% at 90% (was 72%/98%).

**jm-002** (transition; model unchanged — same seed, only the LOO map and metadata changed): medians identical to three decimals. Δ = **+0.81** (50% +0.68 to +0.95; 89% +0.50 to +1.14), P(Δ > 0) > 0.999 — very strong evidence letter sounds track the pure-decoding channel more closely on the ANCOVA parameterisation; slopes β(LS→N) = +1.05, β(LS→W) = +0.24; between-child correlation +0.23 (89% −0.59 to +0.82), direction unresolved.

Power-scaling: jm-002's flags are unchanged from the pre-refit fit (beta_mech[N], Δ, rho_outcome "potential prior-data conflict"); jm-001's slope quantities now read "potential strong prior / weak likelihood" (previously "potential prior-data conflict") — the deliberate regularising Normal(0, 0.3) slope prior making itself felt at n ≈ 53, in both cases a documented sensitivity on an association-only family, not a release blocker.

Both quantities remain **adjusted associations** (latent general ability unblocked); the two Δs are different estimands (levels vs ANCOVA) and their opposite signs are the documented, expected pattern. Artefacts are local (`output/statistical_models/models/`); no upload was run.
