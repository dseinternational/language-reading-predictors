# Adjusting for intervention sessions (IS) in the skill → word-reading mechanism models — collider review and #309 reversal

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Reviewed with, and the decision signed off by, the project owner in session on 2026-07-17. Preliminary.

## Summary

The mechanism models that regress **word reading (WR)** on a single skill exposure were internally inconsistent about whether to adjust for **intervention sessions (IS, the `attend` covariate)**: the letter-sound model **mech-058 (L → W) adjusts it**, but the taught-vocabulary models **mech-088 (TR → W) and mech-089 (TE → W) deliberately omitted it** under the signed-off #309 handling, citing a treatment-affected-collider concern. A formal d-separation check shows the collider concern **does not bite in these models** — because the randomised arm `G` (= the DAG node `IG`) is always conditioned, and that closes the collider path at its arm-fork. Adjusting IS is therefore both **safe** (it opens no new bias) and **necessary** (omitting it leaves a genuine `TR ← IS → WR` confounding path open). We are correcting mech-088 and mech-089 to add IS, aligning them with mech-058, and re-fitting at the reporting tier. This note records the reasoning so the reversal of the #309 decision is auditable.

Scope: only mech-088 and mech-089 change. mech-056 (R → W) and mech-057 (E → W) already carry the correct set (IS is not a parent of R or E, so no `IS` backdoor exists for those exposures — confirmed by the same d-separation search). The GP knee-test variants added on 2026-07-17 (mech-188 TR, mech-189 TE) already include IS. The dose-response (`lrp-rli-dose-*`) and ITT/DiD families are **unaffected** and correctly continue to avoid conditioning on dose / cumulative dose (see "Where #309/#269 still apply" below).

## Why IS is the tricky node — confounder and collider at once

Under the revised base DAG (`dag/dag-language-reading.dagitty`, 2026-07-10), IS has two roles simultaneously in a skill → WR analysis:

- **Confounder.** `IS → TR` / `IS → TE` (more sessions raise taught vocabulary) and `IS → WR` (more sessions raise reading). So `TR ← IS → WR` is a common-cause (backdoor) path that biases the exposure → WR association unless IS is conditioned on.
- **Collider.** `IG → IS` (the randomised arm determines who receives sessions) and `GA → IS` (abler children attend more, latent general ability). So IS is a common _effect_ of the arm and latent ability.

Conditioning on a collider opens an association between its causes. Conditioning on IS therefore opens `IG ↔ GA`, which raises the spectre of the path `TR ← IG → IS ← GA → WR` — the exact concern the #309 handling recorded when it left IS unadjusted.

## The d-separation check

The concern is legitimate in the abstract but has to be evaluated against the _actual_ adjustment set, which always contains the randomised arm `G` (= `IG`, entered as `beta_G`) and age `A`. Using `networkx` d-separation on the backdoor graph (edges out of the exposure removed), with latent general ability `GA` held only hypothetically (it is unobservable, so it can never actually be adjusted — the irreducible residual every mechanism estimate carries):

| Exposure → WR | Current set `Z₀` (no IS) + GA blocks all backdoors? | `Z₁` = `Z₀` + IS, + GA blocks all backdoors? | Does adding IS open any new _observable_ path? |
| ------------- | --------------------------------------------------- | -------------------------------------------- | ---------------------------------------------- |
| TR (mech-088) | **False** — an observable backdoor stays open       | **True** — only latent GA remains            | **No**                                         |
| TE (mech-089) | **False** — an observable backdoor stays open       | **True** — only latent GA remains            | **No**                                         |
| L (mech-058)  | False                                               | **True**                                     | No (already adjusts IS)                        |

The first column is the key finding: **even if GA were observable and adjusted, omitting IS still leaves an open observable backdoor** — and that path is exactly `exposure ← IS → WR`. So the omission is not conservative; it leaves a real confounding leak. Adding IS closes it, after which the _only_ remaining exposure–WR connection is the latent-GA path that can never be closed and is flagged everywhere.

The specific #309 collider path `TR ← IG → IS ← GA → WR`, checked directly:

| Conditioning set                                | `TR ← IG → IS ← GA → WR` open? | Why                                                                                          |
| ----------------------------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------- |
| `{IS}` only (collider conditioned, arm **not**) | **open**                       | the collider IS is activated and the arm-fork IG is free — this is the leak #309 warns about |
| `{IS, IG}` (arm `G` conditioned)                | **blocked**                    | closed at the arm-fork `IG`                                                                  |
| full `Z₁` (the corrected set)                   | **blocked**                    | closed at `IG`                                                                               |

Path shape is what matters: `TR ← IG → IS ← GA → WR` runs _through the arm_, and at `IG` both arrows point outward (`IG → TR`, `IG → IS`), making `IG` a **fork**. A fork is closed by conditioning on it. Because the mechanism models always condition on `G`, the collider path is closed at `IG` regardless of what conditioning on IS does downstream. Conditioning on the collider "opens a door" between arm and ability, but the arm is already held fixed, so the corridor that door opens into is already sealed. You need _both_ the collider activated _and_ the arm free to get the bias; fixing the arm removes it.

## Conclusion

Adjusting for IS in the skill → WR mechanism models is **defensible and correct**: it removes a genuine confounding path (`exposure ← IS → WR`) and, because the randomised arm is always conditioned, introduces none of the collider bias the #309 note feared. mech-058 had it right; mech-088 and mech-089 were the outliers, and we are bringing them into line.

## Where #309 / #269 still apply (unchanged)

The collider concern is real and still governs the families where the **arm or the dose itself is the exposure**, and so is _not_ a conditioned covariate:

- **Dose-response** (`lrp-rli-dose-*`): the exposure _is_ sessions, so there is no arm-fork to close by conditioning; #269's rule — do **not** condition on cumulative prior dose (`attend_cumul`), which reopens the latent-GA backdoor — stands, and cumulative dose remains a flagged sensitivity-only option.
- **ITT / DiD**: the exposure is the randomised arm; conditioning on attendance/dose there would condition on a post-treatment mediator/collider of the arm effect and is correctly avoided.

Only the mechanism family — a _skill_ exposure with the arm as a conditioned covariate — changes here.

## Reproducibility

The check is a `networkx` d-separation script over the base DAG's edge list (backdoor graph = the DAG with edges out of the exposure removed; the "GA held hypothetically" criterion tests whether an observed set blocks every path except the irreducible latent-ability one). The same script reproduces the existing, DAG-reviewed adjustment sets exactly — `L → WR` = {A, HS, IG, IS, SP} (mech-058) and the dose-response `IS → WR` = {A, IG} (dose-077) — which is the validation that it is faithful to the project's earlier #245/#247 derivations.

## Related

- Base DAG: `dag/dag-language-reading.dagitty` (revised 2026-07-10).
- Prior handling being reversed: the #309 taught-vocabulary mechanism decision; see `notes/202607142100-lrp311-taught-vocab-mechanism.md`.
- The dose-response collider rule that remains in force: #269; see `notes/202606291600-period-resolved-lettersound-dose.md` and `notes/202607161800-findings-dose_response.md`.
- Models touched: `lrp-rli-mech-088`, `lrp-rli-mech-089` (re-fit at reporting); findings updated in `notes/202607161800-findings-mechanism.md`.
