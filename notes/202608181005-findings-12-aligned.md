> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings: the `aligned` family — a per-protocol view aligned by when the intervention started

**Read `findings-00-overview` first.** This note covers the 9 models in the `aligned` family. **No quantity in this family is causal**, including the group contrast — which makes it different from every other treatment-adjacent family in the project.

## The data

**RLI trial only**, but arranged differently from anywhere else. Instead of comparing the arms at a common calendar timepoint, this family **aligns each child to their own intervention onset**: the immediate group is measured from timepoint 1 to timepoint 3, the waiting group from timepoint 2 to timepoint 4. Both windows span the same intervention length — about 40 weeks, the full programme.

Each child contributes **one row**: their score at the end of their own window, with their score at their own onset as the baseline. 52–54 children depending on the outcome. There is no child random intercept because there are no repeated rows.

## What the model is for

The `itt` family answers "what happened to children _assigned_ to the intervention over the randomised 20-week window". A reasonable follow-up is: "what did children look like after receiving the _whole_ programme?"

Aligning by onset answers that. It uses the full 40-week exposure for both arms rather than the 20-week randomised window, so it reflects the complete intervention as delivered.

## Why the comparison is not randomised, despite looking like one

This is the crucial point, and it is easy to miss because the model still reports an immediate-versus-waiting contrast.

Randomisation makes two groups comparable **at a common moment**. Aligning by onset destroys that: the immediate group's window runs from timepoint 1 to 3, the waiting group's from timepoint 2 to 4. The waiting-list children are therefore **older** when their window starts, and their window sits later in calendar time, so it may differ in season, schooling, or anything else that changed over the study.

So the cohort contrast here confounds the intervention with age-at-onset and with calendar timing. The models adjust for age at onset and cognitive ability, but adjustment cannot restore a randomised comparison. Every coefficient in this family is an association.

The dose variant adds cumulative sessions as a covariate, which is a **collider** — see the `dose_response` note — so it is a sensitivity view only.

## What was found

| Outcome                 | Cohort contrast (items) | 89% range    |
| ----------------------- | ----------------------- | ------------ |
| Receptive vocabulary    | +2.7                    | −1.8 to +7.2 |
| Letter-sound knowledge  | **+2.2**                | +0.2 to +4.2 |
| Word reading            | +2.1                    | −0.5 to +4.8 |
| Phoneme blending        | +0.3                    | −0.6 to +1.2 |
| Phonetic spelling       | +0.0                    | −0.1 to +0.1 |
| Basic concept knowledge | −0.6                    | −1.7 to +0.5 |
| Receptive grammar       | −1.4                    | −3.1 to +0.3 |
| Expressive vocabulary   | −3.0                    | −6.9 to +0.8 |

Only letter-sound knowledge has an interval clearing zero. Word reading (+2.1) sits close to the randomised estimates from the other families but with an interval spanning zero. The dose sensitivity variant for word reading gives +2.1 [−0.4, +4.8], unchanged.

**The scatter of signs here is the signal to attend to.** Receptive vocabulary comes out at +2.7 and expressive vocabulary at −3.0 — from the same children, over matched windows, on two closely related measures. Both intervals are wide and both include zero. A design giving well-identified effects should not produce that pattern on two such similar outcomes; a design absorbing age and timing differences into its group contrast very well might.

## How to read this family

Treat it as **descriptive corroboration at best**. Where it agrees with the randomised families — letter sounds and word reading positive, at broadly similar magnitudes — that agreement is mildly reassuring, because a badly confounded comparison need not have agreed at all.

Where it disagrees, or produces large opposite-signed vocabulary estimates, the randomised families should be believed and this one should not be used to qualify them.

Note also that this family's intervals are generally **wider** than the `itt` family's despite covering twice the intervention duration. Each child contributes a single row, so there is much less information per child than in the stacked designs.

## What these models cannot tell you

**The cohort contrast is not a treatment effect.** It is confounded by age at onset and calendar timing, by construction.

**"Per protocol" does not mean "better".** Restricting to the delivered programme drops the protection randomisation provides.

**The dose variant conditions on a collider** and is a sensitivity view only.

**Nothing here supports a claim about the full 40-week programme's causal effect**, even though the window is the one that matches the programme as delivered.

## Model inventory

All 9 pass the convergence gate with zero divergences and are publishable: `al-001` (W), `002` (R), `003` (E), `004` (L), `005` (P), `006` (B), `007` (F), `008` (T), and `101` (W, cumulative-session dose variant).
