# Findings review of the current model outputs (dev-config fits, 2026-06-26)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

A careful, model-by-model read of the 17 statistical-model outputs currently in
`output/statistical_models/models/` after the #125 reporting work. The purpose is a
preliminary sense of what each fit says and which findings are solid enough to carry
forward — **not** a final results write-up.

## Read this first — scope and strong caveats

- **All 17 are `dev` config** (500 draws × 2 chains, `target_accept` 0.85). Dev is for
  speed, not inference: tail probabilities (`pd`, `P(benefit ≥ δ)`) carry real
  Monte-Carlo noise and several fits do not meet the convergence gate. **Every number
  below is provisional and must be re-confirmed in `reporting` config before it is
  quoted.** This review is a smoke-test of the *findings*, paralleling the
  smoke-test of the *machinery* in #125.
- **Convergence gate (METHODS.md):** R-hat ≤ 1.01, ESS ≥ 400, divergences = 0,
  BFMI ≥ 0.3. No fit had divergences. The 5 ITT single-outcome models pass cleanly
  (ESS 420–720); the gain/level factor models and the joint mostly read **REVIEW**
  (low ESS — the child-random-intercept models need the longer reporting chains).
  Where a model is REVIEW, I report the *direction* but treat magnitudes and tail
  probabilities as indicative only.
- **Causal discipline.** Only randomised terms are causal: the ITT `tau`, the gain
  family's `beta_trt` (identified by the period-1 randomised contrast), and the level
  family's `b_grp_time[1]` (the t2 contrast). Everything else — own baseline, age,
  ability, cross-skills, mechanism slopes, mediator paths — is an *adjusted
  association* (confounded by latent general ability), reported as "associated with",
  never "drives".
- **Sign convention.** Positive = intervention helps (raises the outcome / the
  probability of coming off the floor).
- **n is small.** ITT singles fit on ~53–54 children (one post-score each); the
  factor/level/DiD/mechanism models stack periods/timepoints (n = rows, not children),
  so their effective information is far smaller than the row count suggests. Expect
  wide intervals and Type-M (winner's-curse) inflation of any large-looking point.

---
## ITT single-outcome models (LRPITT) — all converged (PASS)

The cleanest, most interpretable fits: a single randomised τ per outcome, ESS 420–720.

### LRPITT10 — word reading (W, EWRSWR), n = 53 — strongest standardised signal
- τ (logit) median **+0.355**, 95% CrI **[0.053, 0.650]** (excludes 0); pd **0.991** (≈100:1, *very strong* direction).
- Items scale: **≈ +2.4 words / 79** (95% CrI [0.37, 4.33]); P(benefit ≥ 1 item) = **0.909**, only 9% inside the ROPE.
- **Read:** the most credible single-outcome effect in the suite — the intervention raised standardised word reading, with the interval excluding zero on both scales. Magnitude is modest (~2–3 words). Consistent with letter-sound/word-reading carrying the historical joint signal.

### LRPITT01 — taught receptive vocabulary, block 1 (TR), n = 54
- τ (logit) median **+0.247**, 95% CrI **[−0.013, 0.488]** (just touches 0); pd **0.969** (≈30:1, *moderate*).
- Items: **≈ +1.3 / 24** taught words (95% CrI [−0.07, 2.65]); P(benefit ≥ 1 item) = 0.69 (*inconclusive*), 31% in the ROPE.
- **Read:** positive, direction fairly well supported, but the magnitude is small and the "is it ≥ 1 taught word" question is genuinely uncertain. This is an *intervention-fidelity* outcome (the words were explicitly taught), so a positive direction is expected; the open question is whether the gain clears a meaningful bar.

### LRPITT17 — TR with general-ability (block-design) adjustment, n = 54 — robustness
- τ (logit) median **+0.236**, 95% CrI **[−0.067, 0.513]**; pd **0.955**.
- **Read:** the taught-receptive effect is **robust to adjusting for general ability** — τ barely moves from LRPITT01 (0.247 → 0.236), pd 0.969 → 0.955, interval only slightly wider (the adjustment costs a little precision). The effect is not an artefact of ability imbalance.

### LRPITT09 — phonetic spelling (P, off-floor PRIMARY), n = 53 — credible null
- Off-floor risk difference median **≈ 0** (logit median −0.009, pd **0.488**); **77%** of the posterior mass inside the ±0.10 ROPE.
- Mover counts: intervention **10/28 (36%)** vs control **9/25 (36%)** off the floor — essentially identical by arm. Graded secondary also null (pd 0.37).
- **Read:** no evidence the intervention moved phonetic spelling off the floor; the arms are indistinguishable. With most of the mass inside the ROPE this is "credibly small", not merely "non-significant" — a clean demonstration of the floor-rule + ROPE machinery returning an honest null. (δ = 0.10 is provisional.)

### LRPITT11 — nonword reading (N, off-floor PRIMARY), n = 53 — weak hint
- Off-floor risk difference median **+5.6 pp** (logit median +0.251, 95% CrI [−0.50, 0.98]); pd **0.758** (*suggestive* only); P(RD ≥ 0.10) = 0.31; 66% in the ROPE.
- Mover counts: intervention **12/28 (43%)** vs control **7/25 (28%)** — a ~15 pp *raw* gap that the model shrinks to +5.6 pp with an interval spanning −11 to +22 pp.
- **Read:** a directional hint that the intervention raised the chance of reading *any* nonword, resting on 12 vs 7 children coming off the floor. Far too uncertain to claim on its own at n = 53; flag for the reporting-config refit. (δ provisional.)

### ITT singles — cross-model take
Direction is positive for the taught-vocabulary and reading outcomes (W clearest), null for phonetic spelling, and an uncertain hint for nonword reading. Only **W** has an interval excluding zero. The ability-adjustment robustness check (LRPITT17) is reassuring. Nothing here is "significant-looking but fragile" except possibly N, which is explicitly flagged.

---
## Joint models (LRPITT12, LRPITT15)

### LRPITT12 — all-outcome joint, n = 53 — REVIEW (ESS 353, just under 400)
Per-outcome τ (logit), pd, with intervals that exclude zero flagged:

| outcome | τ median | 95% CrI | pd |
|---|---|---|---|
| **L** letter sounds | **+0.575** | **[0.19, 0.93]** ✶ | **0.997** |
| **W** word reading | **+0.355** | **[0.03, 0.65]** ✶ | **0.984** |
| **TE** taught expressive | **+0.322** | **[0.01, 0.63]** ✶ | **0.976** |
| TR taught receptive | +0.237 | [−0.03, 0.52] | 0.959 |
| B blending | +0.412 | [−0.05, 0.88] | 0.958 |
| UR not-taught receptive | +0.267 | [−0.08, 0.64] | 0.918 |
| P phonetic spelling (graded) | +0.351 | [−0.38, 1.08] | 0.801 |
| UE not-taught expressive | +0.140 | [−0.20, 0.48] | 0.784 |
| R standardised receptive vocab | +0.038 | [−0.13, 0.21] | 0.658 |
| E standardised expressive vocab | −0.003 | [−0.18, 0.16] | 0.482 |

✶ = 95% CrI excludes 0.

- **Pattern:** a coherent gradient — **reading-related and directly-taught skills move (L, W, TE, B, TR); standardised vocabulary does not (R, E ≈ null).** The contrast matrix backs this: L > R and L > E at 0.994, W > R 0.95, W > E 0.98; letter sounds is credibly the largest single effect (L > every other outcome except B/P/W with high probability).
- **Consistency with the singles (the joint's main job):** joint **W τ = 0.355 = the single-outcome LRPITT10 (0.355)**; joint TR 0.237 ≈ single LRPITT01 (0.247). The joint reproduces the single-outcome effects — the cross-model consistency check holds.
- **Caveat:** ESS 353 < 400, so the exact pd values are MC-noisy; the *ordering and the L/W/TE-vs-R/E split* are robust to that, the third-decimal probabilities are not. Refit in reporting config before quoting any single pd.

### LRPITT15 — taught-vs-not-taught expressive generalisation, n = 54 — PASS
- TE (taught expressive) τ **+0.327**, 95% CrI **[0.04, 0.60]**, pd **0.989** — excludes 0.
- UE (not-taught expressive) τ +0.143, 95% CrI [−0.21, 0.50], pd 0.797 — uncertain.
- **Generalisation contrast TE − UE = +0.184**, 95% CrI **[−0.28, 0.63]**, P(diff > 0) = **0.777**.
- **Read:** the intervention credibly raised *taught* expressive vocabulary; the *not-taught* set is uncertain. The taught-minus-not-taught difference is positive (the direction expected if learning is item-specific with limited generalisation) **but its interval spans zero** — at this n we cannot confirm that taught words moved more than not-taught. This, plus the ~null standardised vocabulary in LRPITT12, is the central nuance: **effects concentrate on directly-taught content and reading skills; generalisation to standardised vocabulary is unconfirmed (and looks weak).**

> Note (per project memory): RLI *does* target vocabulary, so the weak standardised-vocabulary result is about **generalisation/measurement** (taught words vs broad standardised tests at small n), not about the intervention "not targeting vocabulary". Do not over-read R/E as "no vocabulary effect".

---
## Gain-factor models (LRPGF) — causal term `beta_trt` (on-intervention, period-1 identified)

`beta_trt` pools every on-intervention period with a child random intercept, so it is
related to but **not identical** to the single-period ITT τ. Everything else is an
adjusted association. **All REVIEW except LRPGF05** (ESS 50–110), so magnitudes are
unreliable here — read direction, not size.

### LRPGF01 — W (word reading), n = 157 rows — REVIEW (ESS 60, very low)
- **Causal `beta_trt`** median **+0.443**, 95% CrI **[0.104, 0.780]**, pd **0.992**; items ≈ **+3.6 / 79** [0.88, 5.89], P(≥1 item) 0.966.
- **Read:** same positive direction as the ITT W effect (LRPITT10 τ 0.355), magnitude a bit larger (+0.443 vs +0.355) — but **ESS 60 means the magnitude is not trustworthy** (Type-M inflation very likely). Direction is the reliable part.
- Associations (not causal): `gamma_own` +0.84 (autoregressive, expected); `gamma_A` **−0.118, pd 0.017** (within-period, older age *associated* with smaller W gain — associational, not "age reduces gains"); `gamma_L` +0.097 (pd 0.978) and `gamma_R` +0.167 (pd 0.895) — letter-sound/vocab "go-together" couplings; all treatment interactions inconclusive.

### LRPGF02 — R (standardised receptive vocab), n = 161 — REVIEW (ESS 50)
- **Causal `beta_trt`** median **−0.032**, 95% CrI [−0.205, 0.131], pd **0.355** — null/indistinguishable from zero; 49% in the ROPE.
- **Read:** no on-intervention effect on standardised receptive vocabulary — consistent with the joint (R pd 0.66, ≈ null). `gamma_own` +0.59 strong; the treatment×ability interaction (pd 0.93) is intriguing but weakly identified at ESS 50 — do not lean on it.

### LRPGF03 — E (standardised expressive vocab), n = 161 — REVIEW (ESS 110)
- **Causal `beta_trt`** median **+0.017**, 95% CrI [−0.136, 0.171], pd **0.572** — null.
- **Read:** no on-intervention effect on standardised expressive vocabulary — consistent with the joint (E ≈ null). Strong `gamma_R` → E association (+0.226, pd 0.997): receptive and expressive vocabulary move together (associational).

### LRPGF05 — P (phonetic spelling, off-floor), n = 159 — PASS (ESS 441)
- **Causal `beta_trt`** median **+0.278**, 95% CrI [−0.50, 1.10], pd **0.75** (suggestive); off-floor risk difference **+2.6 pp**, **98% inside the ±0.10 ROPE**, P(RD ≥ 0.10) = 0.02.
- **Read:** practically null (credibly small — 98% in the ROPE) despite a weakly-positive direction. **Note a discrepancy to reconcile:** the single-period ITT (LRPITT09) gave pd 0.49 (no direction), the GF gives pd 0.75 — the GF pools all on-intervention periods (n 159 vs 53) and the off-floor base rate differs, so the two "on-intervention" estimands are not the same quantity. Both agree the magnitude is negligible.

### LRPGF01b — W treated-only companion, n = 132 — association model (no causal term)
- No `beta_trt` (treated-only ⇒ no randomised contrast). Associations mirror LRPGF01: `gamma_own` +0.785, `gamma_A` −0.121 (pd 0.009), `gamma_L` +0.105 (pd 0.984), `gamma_R` +0.203 (pd 0.923).
- **Read:** purely descriptive of how baseline skills/age co-vary with word-reading gains *among the treated*; the consistent negative age association and positive cross-skill couplings match LRPGF01's adjusted associations. No causal content by construction.

### GF cross-model take
Direction agrees with the ITT/joint: **W positive, R/E null, P negligible.** The
GF `beta_trt` magnitudes run a touch larger than the ITT τ (clearest for W) but the
very low ESS on the graded GF fits makes magnitude comparison premature — this is
exactly what the reporting-config refit is for. The recurring adjusted associations
(autoregressive own-baseline, positive cross-skill couplings, negative age) are
coherent but must never be read causally.

---
## Level-factor models (LRPLF) — causal term `b_grp_time[1]` (the t2 contrast only)

The level model is *not* autoregressive (no own baseline; per-timepoint intercepts
absorb maturation), so its t2 randomised contrast is the same comparison as the ITT
but estimated **without baseline adjustment ⇒ inherently less precise.** Only
`b_grp_time[1]` (t2) is causal; `[0]` is the pre-randomisation t1 timepoint and
`[2]`/`[3]` (t3/t4) are post-crossover associations. **Both LF fits are REVIEW with the
lowest ESS in the suite (40–75)** — magnitudes are unreliable.

### LRPLF01 — W (word reading) levels, n = 210 rows — REVIEW (ESS 75)
- **Causal t2 contrast `b_grp_time[1]`** median **+0.230**, 95% CrI **[−0.22, 0.73]**, pd **0.831**; items ≈ +1.7 / 79 [−1.3, 5.0], P(≥1) 0.66, 30% in ROPE.
- t1 `b_grp_time[0]` −0.198 (pd 0.21) — pre-randomisation, should be ~0; mild chance imbalance, wide. t3/t4 +0.134 (pd 0.73) / +0.060 (pd 0.58) — **the group gap attenuates after crossover**, exactly as the waitlist design predicts (associational once the waitlist is also treated).
- **Read:** same positive *direction* for W as the ITT, but the no-baseline levels view is far less precise (pd 0.83 vs the ITT's 0.99, CrI spans 0). The ITT/GF are the better causal lens for W; the LF's contribution is the trajectory + the visible post-crossover attenuation. Adjusted associations (age, ability×time) are positive and developmentally expected.

### LRPLF02 — R (standardised receptive vocab) levels, n = 215 — REVIEW (ESS 40, very low)
- **Causal t2 contrast `b_grp_time[1]`** median **−0.055**, 95% CrI [−0.25, 0.14], pd **0.299** — null.
- **Read:** null t2 group contrast for standardised receptive vocabulary — consistent with R across the ITT, joint and GF. The model is dominated by **ability×time** associations (`gamma_ability_time[*]` pd 0.996–1.0): receptive vocabulary tracks general ability at every wave (associational, expected). At ESS 40 only the null direction is safe to read.

### LF cross-model take
The t2 causal contrast agrees in direction with the ITT (W positive, R null) but is
the least precise route to it (no baseline). The levels view's real value is
descriptive: the post-crossover attenuation of the W group gap, and the dominant
role of ability in vocabulary levels — both associational. LF ESS is the lowest in
the suite; treat everything here as direction-only pending the reporting refit.

---
## Waitlist-crossover / mechanism / mediation (LRPDID01, LRP56, LRP59)

These three families have no `diagnostics_summary.json` (the convergence banner was
wired into ITT/joint/GF/LF only); convergence read off `diagnostics.csv`.

### LRPDID01 — within-person waitlist-crossover DiD for W, n = 106 rows — marginal convergence
- Convergence: max R-hat 1.015 (intercept), min ESS ~349; the causal `delta` itself is ESS 394, R-hat 1.006 — borderline, refit to confirm.
- **Causal `delta`** (within-person treatment effect) median **+0.370**, 95% CrI **[0.048, 0.697]**, pd **0.986**; ≈ **+2.74 words** [0.37, 5.02]. `beta_period` (maturation anchor) −0.141, pd 0.16.
- **Read — important:** this **replicates the ITT word-reading effect from an independent design.** Between-arm ITT (LRPITT10) gave τ +0.355 / +2.4 words; the within-person crossover gives +0.370 / +2.7 words. Two different identification strategies (between-arm randomisation vs each child as their own control) converge on the same ~+2.4–2.7-word effect. This is the strongest single piece of evidence in the review.

### LRP56 — mechanism dose-response for W, n = 157 rows — marginal convergence
- Convergence: max R-hat 1.01, but the intercept ESS is low (161); the key slope mixes fine.
- `beta_G` = +0.016 [−0.15, 0.19] — **not a treatment contrast** (both arms are on intervention in the pooled phases, so β_G is a pooled coefficient, per the documented caveat). The mechanism curve `f_mech` is shallow and its band spans 0 across the whole mechanism range (e.g. at the top end f ≈ +0.16 [−0.14, +0.67]).
- **Read:** the mechanism→W *adjusted association* is weak and uncertain in dev; nothing to claim. Treat as associational regardless (cross-lagged, same-wave, no measurement-error model). Refit before any interpretation.

### LRP59 — mediation of the W effect through letter sounds, n = 53 — well converged (ESS 450–1640)
- The cleanest-mixing of the three (R-hat ≤ 1.01, ESS ≥ 450).
- g-formula decomposition: **total** +2.9 words (pd 0.996); **NIE (through the mediator)** +1.9 words (pd 0.999); **NDE (direct)** +1.0 words (pd 0.83); **proportion mediated ≈ 0.66**.
- **Read — with a strong caveat:** the *total* effect on word reading is causal (randomised) and again ≈ +2.9 words, consistent with the ITT/DiD. The decomposition *suggests* most of it (~two-thirds) flows through the mediator (letter sounds), with a strong NIE — **but the direct/indirect split rests on the mediator→outcome path, which is an adjusted association, not a causal arrow** (METHODS.md), and it is **unstable to timing**: the t3 sensitivity refit flips the NDE negative (−0.6 words, pd 0.31) and pushes proportion-mediated above 1. So: "consistent with a letter-sound-mediated pathway" is the most that can be said; the precise NDE/NIE split is not a robust causal quantity here.

---
## Cross-cutting synthesis

### 1. Word reading (W) is the robust, replicated effect
Five of six modelling routes agree, three of them well-identified:

| design | estimand | effect (logit) | items | pd | convergence |
|---|---|---|---|---|---|
| LRPITT10 (ITT) | between-arm τ | +0.355 | +2.4 / 79 | 0.991 | PASS |
| LRPDID01 (DiD) | within-person δ | +0.370 | +2.7 | 0.986 | marginal |
| LRP59 (mediation) | total effect | — | +2.9 | 0.996 | PASS |
| LRPITT12 (joint) | between-arm τ | +0.355 | — | 0.984 | REVIEW |
| LRPGF01 (gain) | on-intervention | +0.443 | +3.6 | 0.992 | REVIEW (ESS 60) |
| LRPLF01 (level) | t2 contrast | +0.230 | +1.7 | 0.831 | REVIEW (ESS 75) |

The three well-identified, baseline-adjusted routes (ITT, DiD, mediation-total)
land on **≈ +2.4–2.9 words** with pd ≥ 0.99. The GF estimate is higher but
ESS-unreliable; the LF estimate is lower and less certain because it carries no
baseline. **Two independent identification strategies (between-arm and within-person)
agreeing is the headline.**

### 2. A coherent outcome gradient
From the joint (LRPITT12) and corroborated by the singles/GF/LF:
**letter sounds (L) ≈ word reading (W) ≈ taught expressive (TE) > blending (B) ≈
taught/not-taught receptive (TR/UR) > phonetic spelling/nonword (P/N, floored) >
standardised vocabulary (R, E ≈ null).** The effect concentrates on **reading-decoding
skills and directly-taught content**; **standardised vocabulary does not move** and
**generalisation beyond taught words is unconfirmed** (LRPITT15 TE−UE difference
positive but CrI spans 0). Per project memory, the weak standardised-vocabulary
result is a generalisation/measurement story at small n, not "no vocabulary effect".

### 3. Letter sounds — strongest in the joint, but no single-outcome fit in this batch
L has the largest joint τ (+0.575, pd 0.997) and is the proposed mediator of the W
effect (LRP59 NIE). **But the single-outcome L model (LRPITT07) was not fitted in this
dev batch** — only W, TR, P, N, and TR-with-ability are present as singles. The L
finding currently rests on the joint + the (associational) mediation path. Fit
LRPITT07 in the reporting run to confirm L on its own.

### What is solid vs fragile (dev caveat applies throughout)
- **Solid (direction, replicated):** W positive (ITT + DiD + mediation-total agree, all well-mixed). The R/E standardised-vocabulary nulls (consistent across ITT-joint/GF/LF). The phonetic-spelling off-floor null (LRPITT09, clean PASS, arms identical).
- **Promising but unconfirmed:** L and TE positive (joint only; CrIs exclude 0 there but ESS 353); the letter-sound mediation pathway for W (well-mixed but rests on a non-causal mediator path and is timing-sensitive).
- **Fragile / direction-only:** every GF/LF magnitude (ESS 40–110); the nonword off-floor hint (LRPITT11, pd 0.76); the GF treatment×ability interactions (pd ~0.91–0.93 but ESS ~50).

### Convergence status by family (what to refit in reporting config)
- **Fine in dev:** ITT singles (ESS 420–720), LRP59 mediation (ESS ≥ 450), LRPGF05 / LRPITT15 (PASS).
- **Marginal:** LRPITT12 joint (ESS 353), LRPDID01 (ESS ~349), LRP56 (low intercept ESS).
- **Needs the longer chains most:** all graded GF (ESS 50–110) and both LF (ESS 40–75) — the child-random-intercept models. **Do not quote any GF/LF magnitude until refit.**

### Recommendations for the reporting-config run
1. Run the **full** suite in `reporting` config (not just this 14-model sample) — the single-outcome ITT coverage is partial (missing L/R/E/B/TE/UR/UE singles).
2. Re-confirm the W convergent-validity table and the L/TE joint effects with proper ESS.
3. Reconcile the **GF-vs-ITT discrepancies** (P: GF pd 0.75 vs ITT 0.49; W magnitude GF +3.6 vs ITT +2.4) once GF ESS is adequate.
4. Settle the **provisional ROPE δ** for P/N (and F/T) with the education lead before the floored ROPE cards are quoted.
5. Treat all NDE/NIE and `gamma_*`/`f_mech` quantities as adjusted associations in the write-up, per METHODS.md.
