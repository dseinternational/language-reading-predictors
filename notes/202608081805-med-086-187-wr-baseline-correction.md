<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Decision — correct the MED-086/186 and MED-087/187 baseline-word-reading adjustment

## Decision

Add bare `W` to the adjustment sets and loaded outcomes of MED-086 and MED-087; their parent+100 interventional companions MED-186 and MED-187 inherit the same correction. Bare `W` is baseline word reading $WR_1$. It is distinct from MED-087's `W_pre` structural marker, which the mediation pipeline strips before constructing the confounder set and resolves to the graded outcome's own baseline $B_1$.

This is a measured-backdoor correction, not an identification claim. In the current lagged DAG, the reciprocal edges $WR_1\to LS_2$, $WR_1\to PA_2$ and $WR_1\to NW_2$ make baseline word reading a common cause of the letter-sound mediator and both code-route outcomes. MED-086/186 acquire the open paths $LS_2\leftarrow WR_1\to NW_2$ and $LS_2\leftarrow WR_1\to PA_2\to NW_2$; MED-087/187 acquire $LS_2\leftarrow WR_1\to PA_2$. Conditioning on $WR_1$ blocks these pre-treatment forks. $WR_1$ is not a descendant of randomised group, so adding it does not condition on a treatment-affected variable or create the treatment-induced-confounder problem associated with sessions $IS$.

The decomposition remains non-identified. Latent general ability still confounds the mediator–outcome relation, $IS$ remains a treatment-induced mediator–outcome confounder, and mediator and outcome remain contemporaneous in the primary fit. Adding $WR_1$ reduces one observable source of backdoor bias; it does not turn the natural NDE/NIE or interventional IDE/IIE split into a causal route estimate.

## Parameter-name safeguard

Every existing graded single-mediator model historically names its outcome own-baseline coefficient `b_W`, even when the outcome is not word reading. The new baseline-word-reading outcome-leg confounder therefore uses `b_conf_W`; the mediator leg uses `a_W`. This preserves the established `b_W` geometry and prior role for MED-087, MED-176 and MED-276, while avoiding the existing `b_B` two-mediator blending-path name. The shared naming helper is used by both the factory and the g-formula decomposition. The prior inventory continues to classify `b_W` as the `gamma_own` precision prior and classifies fitted `b_conf_W ~ Normal(0, 0.3)` as a `gamma_cross` association.

## Machine-checked consequences

The current source graph parses as an acyclic 36-node, 197-edge DAG with the six reciprocal $WR_t$ edges `TE`, `TR`, `PA`, `RW`, `LS` and `NW`. The archived mediation derivation passes all model witnesses on the corresponding 53-node, 316-edge three-slice and 70-node, 435-edge four-slice unrolls. A focused d-separation test isolates the new $WR_1$ fork by removing latent general ability and treatment descendants: letter sounds and each outcome are connected without $WR_1$ in the conditioning set and separated when $WR_1$ is added.

Loading the live specifications against the study data gives 50 rows for MED-086/186 and 53 rows for MED-087/187. Reconstructing the previous specifications gives 50 and 54 rows respectively: adding `W` changes the MED-087/187 complete-case sample by one child, while MED-086/186 retain the same 50 rows but gain a fitted confounder in both model legs.

## Stale-output and follow-up rule

Every pre-2026-08-08 numerical result for MED-086, MED-186, MED-087 and MED-187 is stale. That includes `trace.nc`, mediation summaries, sensitivities and calibration artefacts, diagnostics, key findings, rendered reports and any findings-note rows copied from them. The natural/interventional pairs fit the same corrected probabilistic models, so neither member may reuse the parent's old posterior. This correction deliberately does not refit production models inside the code-review PR.

After merge, run fresh `reporting` fits for all four model IDs, require the normal zero-divergence and convergence gates, regenerate every fit-owned artefact, and update the findings notes only from those new trace-backed outputs. Until that follow-up is complete, the four model reports and findings notes must retain a visible stale-results warning and no previous point estimate, interval, direction probability or evidence label should be published.
