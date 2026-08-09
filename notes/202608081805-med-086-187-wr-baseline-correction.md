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

## Stale-output rule and completed follow-up

Every pre-2026-08-08 numerical result for MED-086, MED-186, MED-087 and MED-187 is stale. That includes `trace.nc`, mediation summaries, sensitivities and calibration artefacts, diagnostics, key findings, rendered reports and any findings-note rows copied from them. The natural/interventional pairs fit the same corrected probabilistic models, so neither member may reuse the parent's old posterior. This correction deliberately does not refit production models inside the code-review PR.

The required follow-up completed on 2026-08-09 from merged commit `df00982f53cc44f3635abaef01c4c09ecf185afe`, under the versioned run root `output/runs/20260809T110432Z-df00982f53cc`. Fresh `reporting` fits regenerated every fit-owned artefact for all four model IDs. MED-086 and MED-186 use the same exact 50-child fitted identity (`9b28d8d25a6121504a1b404b72dd5a17f396af7fd47e164edfe1b7993bd686c9`); MED-087 and MED-187 use the same exact 53-child identity (`f3ec82b6ef903f34f5e73fec2a692d6e5e6db6c4dae065395baf1e30c6504a57`). The four primary fits and the separately persisted MED-086 and MED-087 t3 sensitivity subfits all have zero divergences and pass the unrounded R-hat, effective-sample-size and BFMI gates. Each primary release decision is publishable.

The completion discharges the temporary stale-results warning only for these pinned 2026-08-09 artefacts; earlier artefacts remain superseded. The result is deliberately conservative: the corrected off-floor nonword indirect contrast is positive, but its total interval includes zero and the session-dose calibration band extends past the shift where its indirect 89% interval first includes zero; the corrected graded blending indirect interval already includes zero. The current estimates, all sensitivity/calibration details and the verified Azure publication links are recorded in `notes/202608091335-med-wr-baseline-reporting-refit.md`, and the canonical family synthesis is updated in `notes/202608051408-findings-08-mediation.md`.
