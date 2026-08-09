<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Decision — add the lagged `WR → NW` edge, without making a causal claim (#428)

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

## Decision

Adopt `WR_t → NW_{t+1}` as a **provisional working-DAG assumption**. The edge represents word knowledge supporting later nonword decoding through visual analogy or another route that the measured letter-sound and blending nodes do not fully represent. It is an assumption about plausible structure, not a result established by this cohort, and it should be revisited if better longitudinal measurements of decoding routes or general ability become available.

The alternative is not assumption-free. Omitting the edge would assert that every reading-to-later-decoding route is completely mediated by measured letter sounds and blending. That zero-direct-route restriction is too strong given both the independent longitudinal report by Roch and Jarrold (2012; 12 children with Down syndrome followed for four years, with irregular-word reading predicting later nonword reading; [doi:10.1016/j.jcomdis.2011.11.001](https://doi.org/10.1016/j.jcomdis.2011.11.001)) and this cohort's descriptive replication: among children on the nonword floor at t1, 10 of 16 in the upper word-reading half versus 2 of 20 in the lower half had left the floor by t2. Neither observation establishes the direction or mechanism; together they make a structural zero less credible than a qualified edge.

## Identification consequence

Adding the edge does **not** improve backdoor identification. A backdoor graph removes outgoing edges from the exposure, so adding `WR_t → NW_{t+1}` leaves the machine-derived minimal measured sets unchanged: 8 nodes for t1→t2 and 12 nodes for t2→t3. These sets are too wide for n ≈ 54, and latent general ability remains unmeasured. The six-item nonword outcome is also heavily floored. Therefore #433 is unblocked only as an uncertainty-quantified descriptive model: it must retain `causal_status="none"` and `estimand_type="descriptive"`, must not call its association an effect, and must not imply that the edge was learned from these data.

The edge also changes the mediation audit. Together with the already-adopted reciprocal $WR_t\to LS_{t+1}$ and $WR_t\to PA_{t+1}$ edges, it makes baseline word reading a measured common cause of MED-086/186's letter-sound mediator and nonword outcome; the LS and PA reciprocal edges do the same for MED-087/187's blending outcome. Those four specifications now add bare `W`; their pre-correction numerical outputs remain stale, and the required post-merge `reporting` refits completed on 2026-08-09. This blocks the newly recognised measured forks but does not solve latent-general-ability, treatment-induced-session or same-wave mediation limitations. The bounded correction and completed refit are recorded in `notes/202608081805-med-086-187-wr-baseline-correction.md` and `notes/202608091335-med-wr-baseline-reporting-refit.md`.

The graph, its rendered Option-A figure, both archived d-separation assets and the pytest mirror invariant are updated together. The regression test verifies both the direct edge and the unchanged minimal sets; the existing reverse-coupling and mediation d-separation checks remain the acceptance sweep.

## Related decision — optional `WR → LS` LCSM (#429)

Do not build the optional reverse-coupling LCSM now. `lrp-rli-med-176` already answers the direction question over t2→t4, albeit with sensitivity to unmeasured mediator–outcome confounding. A pooled per-transition LCSM would estimate a different quantity, but no present scientific question requires it, and it would inherit rather than solve the same latent-general-ability limitation. Reopen #429 only if the per-transition coupling becomes an explicitly required estimand, not as a hoped-for robustness check on `med-176`.

## Reproducible evidence

- [`202607241600-wr-to-code-lagged-dsep.py`](assets/202607241600-wr-to-code-lagged-dsep.py) derives the 8-node and 12-node sets and checks the established reverse-coupling witnesses.
- [`test_lagged_dag_adjustment_sets.py`](../tests/test_lagged_dag_adjustment_sets.py) mirrors the template into three slices and fails if the source graph, assets and tested structure drift.
- [`202607241600-findings-word-reading-bands.md`](202607241600-findings-word-reading-bands.md) records the descriptive replication and its measurement limitations.

Verification on 2026-08-08: all 61 lagged-DAG tests passed; both archived scripts parsed the 36-node, 197-edge acyclic template and the 53-node, 316-edge three-slice unroll; and the established `WR → TE`, `WR → TR` and mediation d-separation witnesses retained their expected verdicts.

## Reporting-refit completion record — 2026-08-09

The four corrected mediation specifications were fitted at `reporting` tier from commit `df00982f53cc44f3635abaef01c4c09ecf185afe` under `output/runs/20260809T110432Z-df00982f53cc`. The natural/interventional pairs have exact matching fitted-child identities: 50 children for MED-086/186 and 53 for MED-087/187. All four primary fits and both required natural-model t3 subfits pass the automatic zero-divergence, R-hat, effective-sample-size and BFMI gates, with trace-backed provenance for each t3 subfit. This computational result completes the refit obligation but does not strengthen the edge into an empirical or causal claim. The resulting off-floor indirect contrast is positive but prior/model/confounding sensitive and accompanies a total interval crossing zero; the graded blending indirect interval also crosses zero.
