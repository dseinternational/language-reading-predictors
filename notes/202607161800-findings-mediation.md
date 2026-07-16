# Findings — mediation & mediation_multi families (how the reading gain is carried)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)). Preliminary. (Covers both the single-mediator `mediation` family and the two-mediator `mediation_multi` family.)

## What these models ask

Given that the intervention raises word reading (from the [ITT](202607161800-findings-itt.md) analysis), _through which skill does that effect travel_? These models use a **g-formula decomposition** by counterfactual simulation to split the **total** effect into:

- **NDE / direct** — the part **not** running through the named mediator;
- **NIE / indirect** — the part that **does** run through the mediator (e.g. "the programme taught letter sounds, and letter sounds carried the reading gain").

(`med-078/186/187` are "interventional-effects" companions — a slightly different formal interpretation of the same split, labelled IIE.) Everything is on the **items scale** (percentage points for the floored nonword outcome).

**Crucial caveat, stated in every model:** the direct/indirect split is **model-based, not an identified natural effect**. It assumes there is no unmeasured confounder sitting between the mediator and the outcome. So NIE/NDE are **best read as a plausible decomposition under assumptions**, not as proven pathways. `med-079` is a deliberate **negative control** (a route through grammar that the causal diagram says should carry ≈0) used to calibrate how much "leakage" residual confounding produces.

## Convergence gate

All 16 models (13 single-mediator + 3 two-mediator) **passed**.

## Headline decompositions

| Model       | Route                                               | Total              | Indirect (NIE)                | Read                                                                             |
| ----------- | --------------------------------------------------- | ------------------ | ----------------------------- | -------------------------------------------------------------------------------- |
| **med-059** | W via **letter sounds (L)**                         | +1.9 items (93% +) | **+1.7 items** (+0.4 to +3.6) | Most of the word-reading gain travels via letter sounds.                         |
| **med-076** | W via L at t2 → reading at t4 (longitudinal)        | +2.4 items (89% +) | **+3.1 items** (+0.9 to +6.0) | Same story with correct time-ordering: the indirect L route is clearly positive. |
| med-064     | W via **L vs expressive vocab (E)** (two mediators) | +2.0 items (91% +) | +2.0 joint (+0.3 to +4.4)     | The joint indirect route is clearly positive; L dominates E.                     |
| med-066     | W via **L vs blending (B)** (two mediators)         | +2.9 items (99% +) | +1.6 joint (+0.1 to +3.5)     | Code-route mediators jointly carry a positive share.                             |
| med-075     | Sequential **L → B → reading**                      | +2.9 items (99% +) | +1.6 joint (+0.2 to +3.4)     | Consistent with a decoding chain.                                                |
| med-062     | W: code-route (L+B) vs lexical share                | +1.6 items (89% +) | +0.9 (−0.1 to +2.6)           | Indirect share positive-leaning but less certain.                                |
| med-068     | W via **taught expressive vocab (TE)**              | +2.5 items (99% +) | +0.6 (−0.3 to +2.0)           | Small, uncertain vocab route.                                                    |
| med-080     | W via **taught receptive vocab (TR)**               | +2.4 items (98% +) | +0.4 (−0.3 to +1.9)           | Small, uncertain vocab route.                                                    |
| med-087/187 | **Blending (B)** via L                              | +0.7 items (93% +) | +0.2 (−0.1 to +0.7)           | Small L→B indirect share.                                                        |
| med-086/186 | **Nonword (N, off-floor)** via L                    | +7.9 pp (88% +)    | **+8.7 pp** (+1.7 to +17.8)   | The off-floor nonword move runs strongly via letter sounds.                      |
| med-074     | W via **nonword decoding (N)**                      | +2.7 items (99% +) | +0.1 (−0.3 to +0.8)           | Nonword route ≈ 0 (floor-limited).                                               |
| **med-079** | W via **grammar (T)** — negative control            | +2.2 items (97% +) | **+0.1 items** (−0.3 to +0.8) | The DAG-severed route reads ≈0, as it should — good calibration.                 |
| med-092     | W via L (period-stacked scaffold)                   | +3.0 items (99% +) | +0.8 (+0.2 to +1.6)           | Consistent positive L route.                                                     |

## The one-paragraph story

The word-reading benefit is **most consistently carried by letter-sound knowledge**: the indirect route through L is clearly positive in several independent framings (`med-059` +1.7 items; longitudinal `med-076` +3.1 items; `med-092` +0.8), and the same is true for the off-floor **nonword** gain (`med-086` +8.7 pp via L). Routes through **vocabulary** (taught or standardised) carry only a small, uncertain share. The **negative-control** route through grammar (`med-079`) correctly reads ≈0, which tells us residual confounding is not manufacturing large spurious indirect effects. This lines up with the whole-study picture: **the programme works on reading largely by building the code (letter sounds → decoding), not by broadly lifting vocabulary.**

## What is causal

The **total** intervention contrast inherits the randomised ITT logic. The **direct/indirect split is not identified** — it is a decomposition under a no-unmeasured-mediator-outcome-confounding assumption, so read NIE/NDE as "under this model" and lean on the negative control (`med-079`) and the sensitivity sweeps when weighing them.
