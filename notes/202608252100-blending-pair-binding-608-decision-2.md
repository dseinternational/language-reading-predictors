<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# #608 decision 2, amended: bind the run plan, not the archive

- **Date:** 2026-08-25
- **Status:** scientific decision — **amends #608 decision 2**
- **Issue:** #619 / #608

## The decision

**Supersedes** #608 decision 2 ("Every pair is bound by the content-addressed archive apparatus, not only the ITT pair. The local two-directory pair check is retired as a permanent arrangement.").

In its place:

1. **Every pair binds its resolved run plan.** The two halves must resolve identical plans apart from the score-mean link and the pairing bookkeeping derived from it. A link sensitivity that also moved a term, a prior or an adjustment set is not a link sensitivity.
2. **Every half must be current.** Each stored plan must still match what its own module resolves today. A fit that predates a change to its own specification cannot be released, whether or not its twin is equally stale.
3. **Provenance is recorded and surfaced, not required to match.** Source commit, dirty flag and environment-lock hash go in the pair's cards, and a plain-language note says when the halves differ.
4. **The content-addressed archive stays ITT-only** — the pair published outside this repository.

## Why the original decision was the wrong instrument

#608 named two weaknesses of the stored-artefact check: it cannot detect that one half was fitted from different source, and a reader cannot see which evidence tier backs a card. Generalising the archive was the proposed fix. Working through it, three things came out.

**The archive's cost is real and its benefit is elsewhere.** Generalising it means duplicating **1.93 GB** of trace files across the seven non-ITT pairs (more once every companion exists); the ITT archive alone is already 106 MB. What that buys is immutable, independently re-derivable evidence — genuinely valuable for a fit published outside the repository, which is why ITT keeps it. It does not detect a mismatched specification, which is the failure mode that has actually occurred here, twice, in a week.

**"Same source" is not attainable and was never the right test.** A companion is registered in a _later_ commit than its primary by construction, so the two halves cannot share a commit unless the primary is refitted afterwards. Of the six stored-artefact pairs, **five** were fitted at different commits and **four** under different environment locks (#617's dependency bump landed between them). Requiring either would fail-close most of the policy for a fact about git history rather than about the models. What actually establishes that two fits are the same analysis is that they resolve the same run plan — which is checkable, cheap, and true of all six today.

**The check the archive would have brought is the recomputation, not the hashing.** Both defects below were found by asking "does this stored artefact still match what the code produces?", never by comparing bytes.

## The two defects that settled it

**`lrp-rli-lf-006` published an estimand the repository had superseded.** Its stored card was +0.6449; current code gives +0.6375. The fit dates from commit `4e924948` (2026-08-19), four days before #594 changed `level_t2_marginal_effect` to net out the balance term (#584 decision 1). Confirmed by reproducing both values: calling the current function with `balance_term=None` returns the stored number exactly. The refit also picked up #598's dispersion and child-SD priors, which the stored fit predated — so it was stale in three ways at once, and nothing in the pipeline said so.

**A `mediation` pair was published comparing two different models.** The stored `lrp-rli-med-087` predated #600 and so lacked `a_base_B` and `b_base_L`, the per-leg baseline terms #585 established the g-formula requires. Its companion, built during #619, carried them. Every check that existed passed — data, fitted rows, sampling configuration, card shape — and the published NDE/NIE comparison was confounded between the link and the leg contract. It was found by diffing each stored plan against what its module resolves today, and the numbers were corrected before the pair was merged. The confounding turned out small (the stale primary gave total +0.674 against the refit's +0.677), so the finding survived; the check is what was missing, not the conclusion.

Both are caught by the amended binding. Regression tests assert exactly that, against the two real stored fits rather than synthetic perturbations, and a mutation check confirms they fail when the diff is disabled.

## What "current" costs

Requirement 2 makes a stored run plan a release input, which means a change to a family's resolver invalidates that family's stored fits until they are refitted. That is the point — a fit whose plan the code no longer produces is a fit whose published numbers nobody has verified — but it is friction, and it should be understood as deliberate rather than discovered later. It also means the refit backlog is now partly _visible_: `lrp-rli-did-003` shows real #613 drift and `lrp-rli-itt-008` / `108` differ in `kappa_prior_family`, so those pairs will fail this check until refitted.

Two false positives were designed out on the way, both of which would have made the check noise rather than signal:

- **Serialisation.** Resolvers produce tuples; `config.json` returns lists. An early draft compared them raw and flagged fits made minutes earlier. `_normalise_plan_value` is that fix, and a test pins it.
- **Declaration style.** `settings_source` records whether settings came from a typed dataclass or the legacy `extra` dict. Two fits whose resolved settings agree are the same model either way, so comparing it would fail a pair whose companion was written in the newer style. Excluded, with the reason recorded in the code.

## What this does not do

It does not recompute each family's published card from its trace. That was in the proposal this note implements and is **not** in it — the run-plan and currency checks caught both known defects, and per-family recomputation is a materially larger piece (`concurrent` does not persist the term scales its `+k items` rows need; `dose_response` would need its contrast rebuilt from `dose_support.csv`; `mediation` would mean re-running a stochastic 50-replicate g-formula inside a validator). It remains the natural next increment if a defect appears that changes a card without changing a plan. Worth noting the #619 link fixes were nearly that class — they changed reporting helpers — but they also added plan fields, so this check would have caught them.

It also leaves the ITT archive exactly as it is, including its `dirty=False` requirement. Nothing about ITT changes.
