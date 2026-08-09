<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted and edited by LLM-based AI tools (Claude Code/Opus 5 and Codex/GPT-5).

# Findings 08 — the mediation family (what carries the effect)

> [!WARNING]
> **Stale MED-086/186 and MED-087/187 rows (2026-08-08).** The current lagged DAG exposes baseline-word-reading forks into the letter-sound mediator and both code-route outcomes. Those four specifications now adjust for baseline word reading and require fresh `reporting` fits. Their numerical rows, the prose interpreting the 8.7-percentage-point and 0.23-item indirect effects, and any family synthesis that relies on them are historical only and must not be published or interpreted until replaced from new trace-backed outputs. The other mediation models in this note are unaffected by this correction; see `notes/202608081805-med-086-187-wr-baseline-correction.md`.

Reports every model in the `mediation` family (15) and the `mediation_multi` family (4) from the 2026-08-04/05 `reporting` refit. **19 models, all passing the convergence gate.** Reading conventions in note 00. Preliminary research data — all estimates provisional.

## What these models do

The ITT family shows the intervention raises word reading. This family asks **through what**. If the programme teaches letter sounds and letter sounds are used to read words, the word-reading gain should run through the letter-sound gain rather than around it.

**Design.** A g-formula decomposition by counterfactual simulation. The model is fitted, then the posterior is used to simulate what would have happened under combinations of treatment assignment and mediator value, splitting the total effect into:

- **NDE** (natural direct effect) — the part that would remain if the mediator did not respond to treatment;
- **NIE** (natural indirect effect) — the part carried _through_ the mediator;
- **total** = NDE + NIE.

Several models also report the **interventional-effects** version (IDE/IIE), which targets a slightly different and more robustly-defined estimand under interference between mediators.

**What this family can and cannot claim.** The _total_ effect inherits the ITT's randomisation warrant. The **split** between direct and indirect does not: decomposing a total effect requires assuming there is no unmeasured confounding of the **mediator–outcome** relationship, and general ability plausibly confounds letter sounds and word reading together. Randomisation balances the arms; it does not make the mediator exogenous. So the NIE is a **model-based decomposition under an untestable assumption**, not a randomised quantity. Everything below should be read at that strength.

All quantities are on the word-reading items scale unless marked otherwise.

## The headline: the word-reading gain runs through letter sounds

| Model     | Mediator                                    | Total | NDE (direct)   | NIE (indirect)             |  P(NIE>0) |
| --------- | ------------------------------------------- | ----- | -------------- | -------------------------- | --------: |
| `med-059` | Letter sounds                               | +1.95 | +0.16 (P=0.55) | **+1.69** (+0.58 to +3.22) | **0.997** |
| `med-078` | Letter sounds (interventional)              | +1.95 | +0.15 (P=0.55) | **+1.70** (+0.58 to +3.24) | **0.996** |
| `med-076` | Letter sounds at t2 (longitudinal ordering) | +1.39 | −1.68 (P=0.18) | **+2.92** (+1.12 to +5.32) | **0.997** |
| `med-092` | Letter sounds (period-stacked)              | +3.03 | +2.24 (P=0.97) | +0.75 (+0.29 to +1.39)     |     0.999 |
| `med-062` | Code-based reading route (composite)        | +1.62 | +0.61 (P=0.68) | +0.92 (+0.06 to +2.17)     |     0.959 |

**In the cross-sectional models essentially the whole effect is indirect.** `med-059` puts the direct effect at +0.16 items with a direction probability of 0.55 — indistinguishable from nothing — while the route through letter sounds carries +1.69 with very strong evidence. The interventional-effects version agrees to two decimal places. The longitudinal-ordering model, which requires the mediator to be measured before the outcome, gives an even larger indirect effect.

`med-092` is the exception worth noting: stacking every period, the direct effect is +2.24 and the indirect +0.75. That model uses far more rows and a different treatment definition ("on the programme" rather than assigned), so it is not directly comparable, but it is a reminder that the "almost all indirect" split is strongest in the single-window models.

## Which routes do _not_ carry the effect

This is where the family earns its keep — several plausible-sounding routes come out empty.

| Model     | Mediator                                 | NIE                    | P(NIE>0) | Reading               |
| --------- | ---------------------------------------- | ---------------------- | -------: | --------------------- |
| `med-074` | Nonword decoding                         | +0.02 (−0.41 to +0.50) |    0.555 | nothing               |
| `med-079` | Receptive grammar (**negative control**) | +0.08 (−0.19 to +0.63) |    0.713 | nothing — as designed |
| `med-080` | Taught receptive vocabulary              | +0.21 (−0.30 to +1.23) |    0.754 | weak at best          |
| `med-068` | Taught expressive vocabulary             | +0.30 (−0.23 to +1.32) |    0.814 | weak at best          |

**The negative control behaves.** `med-079` routes the effect through receptive grammar — a skill the intervention has no mechanism to use for word reading — and finds nothing (NIE +0.08, inconclusive), with the direct effect absorbing the total. That is exactly what a well-behaved negative control should do, and it is meaningful evidence that the letter-sound result is not an artefact of the decomposition machinery producing large indirect effects for any mediator handed to it.

**Nonword decoding does not mediate, which is initially surprising.** The mechanism family (note 07) found letter sounds → nonword reading to be the strongest coupling in the suite. Yet routing the _intervention effect_ through nonword decoding gives NIE +0.02 with the direct effect taking +2.83. The two facts are compatible: letter-sound knowledge is strongly associated with nonword decoding, but the intervention's effect on word reading does not travel through a measurable nonword-decoding step. The sequential model `med-060` confirms it directly — running the route letter sounds → nonword decoding → word reading gives NIE_L +1.66 (P = 0.989) and **NIE_N −0.07 (P = 0.31)**.

## The two-mediator models

These fit two mediators at once, so the routes compete for the same effect rather than being estimated in separate models.

| Model     | Mediators                                     | NIE via letter sounds | NIE via the other |
| --------- | --------------------------------------------- | --------------------- | ----------------- |
| `med-064` | Letter sounds + expressive vocabulary         | **+1.88** (P=0.996)   | +0.03 (P=0.58)    |
| `med-066` | Letter sounds + phoneme blending              | **+1.61** (P=0.995)   | −0.03 (P=0.42)    |
| `med-075` | Letter sounds + phoneme blending (sequential) | **+1.62** (P=0.995)   | −0.04 (P=0.42)    |
| `med-060` | Letter sounds + nonword decoding (sequential) | **+1.66** (P=0.989)   | −0.07 (P=0.31)    |

**The letter-sound route survives every competitor.** Put head to head with expressive vocabulary, phoneme blending or nonword decoding, letter sounds carry +1.6 to +1.9 items with very strong evidence and the alternative route carries nothing. This is the most robust mechanistic finding in the suite.

## Other outcomes — four rows awaiting corrected refits

| Model                 | Outcome via letter sounds               | NIE                       | P(NIE>0) |
| --------------------- | --------------------------------------- | ------------------------- | -------: |
| `med-086` / `med-186` | Nonword reading (off-floor probability) | +0.087 (+0.028 to +0.160) |    0.993 |
| `med-087` / `med-187` | Phoneme blending                        | +0.23 (−0.03 to +0.59)    |    0.921 |

The numerical rows above are retained only as a record of the superseded fits. No current conclusion about the letter-sound route to nonword reading or phoneme blending should be drawn until MED-086/186 and MED-087/187 have been refitted under the corrected baseline-word-reading adjustment.

## Could the causality run the other way?

Two models reverse the ordering — does word reading carry the effect to letter sounds, rather than the reverse?

| Model     | Route                              | NIE                    | P(NIE>0) |
| --------- | ---------------------------------- | ---------------------- | -------: |
| `med-176` | Word reading at t2 → letter sounds | +0.45 (+0.04 to +1.11) |    0.963 |
| `med-276` | Same, t3 outcome (less ceilinged)  | +0.50 (+0.05 to +1.23) |    0.966 |

**The reverse route is not empty.** It is roughly a quarter the size of the forward route (+0.45 against +1.69) and carries moderate rather than very strong evidence, but it does not vanish. This is an honest limit on the mediation story: with these data the forward direction is clearly the larger, but reverse mediation cannot be ruled out, and both would be consistent with a shared underlying process driving letter sounds and word reading together.

## Caveats

- **The decomposition is not randomised.** Only the total effect inherits the ITT warrant. The direct/indirect split assumes no unmeasured mediator–outcome confounding, and latent general ability is a plausible violation.
- **`proportion_mediated` is not quoted.** It is a ratio of two uncertain quantities whose denominator's interval spans zero for most of these models, so it is unstable and easily over-read. The NDE and NIE with their own intervals are the honest summary.
- **PSIS-LOO is not computed** for this family — the g-formula fits are simulation-based decompositions rather than predictive models, so the usual model-comparison route does not apply.
- **The two-mediator models are the slow ones** (`med-064/066/075` take 36–41 minutes each at reporting tier) because the sensitivity sweep runs 42 full counterfactual decompositions over all 36,000 draws.
- **Reverse mediation is not excluded** — see above.
- **Predictive calibration.** 50% bands cover about 71% of observations (79% for the two-mediator models).

## Where this leads

Taken with note 07, the still-supported mechanism picture is: the intervention raises letter-sound knowledge; letter-sound knowledge carries essentially the whole word-reading gain; no vocabulary or blending route competes with that word-reading result; a grammar negative control correctly shows nothing; and the reverse ordering, while smaller, is not excluded. The former claim about the whole off-the-floor nonword-reading gain is withheld pending the corrected MED-086/186 refits, and the MED-087/187 blending decomposition is likewise withheld.
