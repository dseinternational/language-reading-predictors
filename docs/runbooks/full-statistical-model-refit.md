<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).
>
> Substantially edited in the waitlist-crossover guidance by a LLM-based AI tool (Codex/GPT-5).

# Runbook — full statistical-model refit, render, publish and record

A start-to-finish checklist for refitting **every registered Bayesian statistical model** (the Layer-2 PyMC families) at reporting quality, rendering the reports, publishing them to the public research site, and recording the run. Follow it top to bottom; each step says how to check it worked before moving on.

This runbook is the operational companion to the [`lrp-fit-statistical`](../../.claude/skills/lrp-fit-statistical/SKILL.md) skill (which covers the _science_ — estimands, the convergence gate, how to read each family) and to `METHODS.md`. It captures the **workflow and the gotchas** — the things that are not obvious from the scripts and that cost time the first time round. A worked example of its output is `notes/202607131300-full-statistical-refit-reporting.md`.

**Time budget:** a full `reporting` sweep of 115 models + renders is roughly **1.5–2 h on 16 cores**. Most fits are fast (a typical ITT fit is ~40 s under `nutpie`); the slow tail is the growth / mediation / HSGP / LCSM / `corr_factor` models. Plan to run the fit step in the background.

---

## Step 0 — Prerequisites

Do all of these once, before you start. A failure here is the most common reason a later step dies halfway.

```bash
# 1. Activate the environment (the Bash-tool shell has NO conda env active by default).
conda activate dse-language-reading-predictors

# 2. Verify the scientific core matches the canonical spec.
dse-check-env environment.yml

# 3. Authenticate to Azure with an account that has the write role on the
#    public `dseresearch` container (Frank's `az login` identity does).
az login          # run interactively in your own terminal (prefix with `!` in Claude Code)
az account show   # confirm the right tenant/subscription

# 4. Make sure Quarto is on PATH and points at the env Python.
export PATH="/Applications/quarto/bin:$PATH"
export QUARTO_PYTHON="$(python -c 'import sys; print(sys.executable)')"
quarto --version
```

Decide **where output goes**. The default is repo-local `output/`. For a VM run with a scratch disk, redirect it — but remember scratch disks are ephemeral, so you must upload durable artefacts before teardown:

```bash
export DSE_LRP_OUTPUT_DIR=/mnt/scratch/lrp   # optional; resolution lives in src/.../paths.py
```

The resolved output root is printed at the start of each long-running command and recorded in every `config.json`.

---

## Step 1 — Fit every model (`reporting`)

```bash
python scripts/fit_statistical_model.py all --config reporting --render
```

- `all` fits every registered Layer-2 model sequentially; `--config reporting` = 6 chains × 6000 draws × 6000 tune, `target_accept = 0.95`. `--render` renders each model's Quarto report **in a batch after all fits finish** (see the gotcha below).
- Run it **in the background** with an output log so you can watch progress and survive a disconnect:

```bash
nohup python scripts/fit_statistical_model.py all --config reporting --render \
  > /tmp/refit.log 2>&1 &
tail -f /tmp/refit.log
```

- **Check the exit code, not just the "N fitted, 0 failed" line.** The script exits non-zero if any fit **or** render failed, **or** if the built-in `--upload` step raised. An upload 403 (see Step 5) makes it exit 1 even when every fit and render succeeded — so a non-zero exit does not always mean the fits are bad. Inspect the log.

> [!IMPORTANT]
> **Gotcha — `--render` is batched, not per-model.** `fit_statistical_model.py all --render` runs its Quarto render loop **once, at the very end**, after every model has fitted, iterating an in-memory list built during that process's fit loop. If the sweep is **interrupted** (killed, crashed, disconnected) partway, the completed model dirs have `trace.nc` + CSVs + `config.json` but **no `index.html`** — nothing was rendered. See Step 3 for how to render already-fitted dirs without re-fitting.

### If the sweep was interrupted: resume, don't restart

Do **not** re-run `all` from scratch — completed fits are intact and expensive. Re-fit only the models whose output dir is missing the fit artefacts (`diagnostics_summary.json` + `trace.nc`):

```bash
# List model dirs that fitted vs. those still missing the gate file.
for d in output/statistical_models/models/*-reporting; do
  [ -f "$d/diagnostics_summary.json" ] || echo "MISSING: $(basename "$d")"
done
```

Then fit just those, by id:

```bash
python scripts/fit_statistical_model.py lrp-rli-mm-001 --config reporting --render
```

(In the worked example, the first sweep was killed at 112/115; only `lrp-rli-mm-001`, `lrp-rli-mm-101`, `lrp-rlm-hg-001` were re-fitted.)

---

## Step 2 — Verify the fits and triage the convergence gate

**Confirm all models fitted.** There are 115 registered models; the count must match:

```bash
ls -d output/statistical_models/models/*-reporting | wc -l   # expect 115
```

**Read the gate.** Each `diagnostics_summary.json` has `passed` + `checks` = {rhat, ess, divergences, bfmi}. Thresholds: **r̂ ≤ 1.01, ESS ≥ 400, BFMI ≥ 0.30, zero divergences** (the divergence check fails on _any_ divergence). A quick sweep of what passed and what to look at:

```bash
python - <<'PY'
import glob, json, os
for d in sorted(glob.glob("output/statistical_models/models/*-reporting")):
    try:
        s = json.load(open(f"{d}/diagnostics_summary.json"))
    except FileNotFoundError:
        print(f"NO GATE {os.path.basename(d)}"); continue
    if not s["passed"]:
        print(f"FLAG {os.path.basename(d)}: div={s['divergences']} "
              f"rhat={s['max_rhat']:.4f} ess={s['min_ess']:.0f}")
PY
```

**Triage — do not silently accept or silently reject flags.** Expected, usable flags:

- **Divergence-only, ≤ ~0.5 % of draws** (GP/HSGP mechanism surfaces, dose slopes, horseshoe funnel): within the METHODS ≤ 1 % guidance — **usable**, note them.
- **A genuine concern** looks like `mm-001` (`corr_factor`): a latent-factor funnel with a mild r̂/ESS miss. Its **correlations are robust** (the deliverable); its **structural leg is read cautiously**. Hold structural coefficients pending a non-centred reparameterisation or higher `--target-accept`.

Confirm **no headline causal estimand sits on a flagged model** before publishing. `--target-accept 0.97` (or `0.99`) is the first lever to chase away divergences on a hard geometry.

---

## Step 3 — Render reports (only if Step 1 didn't, or a resume left gaps)

If `--render` completed in Step 1, skip this. If the sweep was interrupted (so renders never ran) or a resume added dirs, render the already-fitted dirs **standalone** — no re-fit. Each dir carries its own CSVs and copied partials, so `quarto render index.qmd` inside it is self-contained.

Save this as `render_all.sh` and run it (a 4-way bash loop; **do not** use `xargs -I{}` with a multi-line `sh -c` template — BSD xargs dies with "command line cannot be assembled, too long"):

```bash
#!/bin/bash
cd "$(git rev-parse --show-toplevel)" || exit 1
export PATH="/Applications/quarto/bin:$PATH"
export QUARTO_PYTHON="$(python -c 'import sys; print(sys.executable)')"
RLOG=/tmp/render_all.log; : > "$RLOG"

render_one() {
  local d="$1"
  ( cd "$d" && quarto render index.qmd >/dev/null 2>render.err )
  [ -f "$d/index.html" ] && echo "OK $(basename "$d")" >>"$RLOG" \
                         || echo "FAIL $(basename "$d")" >>"$RLOG"
}

MAXJOBS=4; count=0
for d in output/statistical_models/models/*-reporting; do
  [ -f "$d/index.html" ] && continue          # skip already-rendered
  render_one "$d" &
  count=$((count+1)); (( count % MAXJOBS == 0 )) && wait
done
wait
echo "DONE OK=$(grep -c '^OK' "$RLOG") FAIL=$(grep -c '^FAIL' "$RLOG")" >>"$RLOG"
cat "$RLOG" | tail -3
```

**Verify HTML coverage before uploading** (missing HTML → published dirs with no report page):

```bash
for d in output/statistical_models/models/*-reporting; do
  [ -f "$d/index.html" ] || echo "NO HTML: $(basename "$d")"
done
```

---

## Step 4 — Cross-model comparison

```bash
python scripts/compare_statistical_models.py --config reporting
```

Writes to `output/statistical_models/comparison/`: `itt_vs_joint_tau.csv` (single-outcome τ vs joint τ_k consistency), `tau_forest.png`, `mediation_family.csv` + `mediation_family_forest.png`, and nested PSIS-LOO tables (mechanism / phonics-route / age-moderation / dose / did-dose). The script must exclude gate-flagged models from interpretable LOO tables and verify identical observation identities and order before comparing models. Note every artefact it skips and why; equal row counts alone are not sufficient evidence that pointwise LOO values are aligned.

---

## Step 5 — Publish to the public research site

> [!IMPORTANT]
> **Public + preliminary — confirm the scope with the study owner before publishing.** These reports become anonymously readable on the internet.

> [!IMPORTANT]
> **Gotcha — the built-in `--upload` 403s from a VM.** The public `dseresearch` container needs the write role. `az login` (Frank's identity) has it; the VM **managed identity** does not, and `DefaultAzureCredential` prefers the MI — so the built-in flag fails with 403. Publish with the small wrapper below, which forces `AzureCliCredential`, and run it with `AZURE_CLIENT_ID` unset so nothing re-selects the MI. To make `--upload` first-class instead, grant the runner identity **Storage Blob Data Contributor** on the `dseresearch` account.

Save as `upload_public.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
import glob, os, uuid
from azure.identity import AzureCliCredential
from dse_research_utils.storage.azure import upload_directory_to_blob_storage

credential = AzureCliCredential()
run_id = str(uuid.uuid7())            # ONE run_id for the whole batch
print(f"RUN_ID={run_id}", flush=True)

targets = sorted(glob.glob("output/statistical_models/models/*-reporting"))
targets.append("output/statistical_models/comparison")

ok, fail = {}, []
for d in targets:
    name = os.path.basename(d)
    try:
        res = upload_directory_to_blob_storage(
            d, name, project="language-reading-predictors",
            include_traces=False,      # traces are ~16 GB across the suite
            run_id=run_id, credential=credential)
        ok[name] = res.report_url
        print(f"OK {name} -> {res.report_url}", flush=True)
    except Exception as exc:           # noqa: BLE001
        fail.append((name, repr(exc)))
        print(f"FAIL {name}: {exc!r}", flush=True)

print(f"\nUPLOADED={len(ok)} FAILED={len(fail)}  RUN_ID={run_id}")
```

Run it with the MI de-selected:

```bash
unset AZURE_CLIENT_ID
python upload_public.py
```

**Verify public access** on a representative report (expect `200 text/html`):

```bash
curl -s -o /dev/null -w "%{http_code} %{content_type}\n" \
  "https://dseresearch.blob.core.windows.net/public/projects/language-reading-predictors/output/<RUN_ID>/lrp-rli-itt-010-reporting/index.html"
```

The report root is `https://dseresearch.blob.core.windows.net/public/projects/language-reading-predictors/output/<RUN_ID>/`; per-model reports are `<root>/<model-id>-reporting/index.html`.

**Private durable archive (alternative / additional).** If you need a durable copy and only have the managed identity, `azcopy` to the private `outputs` container works today (see the `lrp-fit-statistical` skill, "Upload → B"). Always exclude traces (`--exclude-pattern="*.nc"`) and mask the account/host in any displayed output.

---

## Step 6 — Record the run in a dated note

Every full refit gets a dated `notes/` note. Follow `METHODS.md` "Interpret / Reporting results" and the #179 evidence ladder. Use the AI-authorship label if the note was AI-drafted.

**Pull the headline numbers programmatically** rather than by eye — one script that reads each family's result CSV (per the file→column map in the `lrp-fit-statistical` skill) keeps the note accurate. A reusable extractor lives in the worked-example session; the essentials per family:

| Family                 | File                                                        | Read                                                                                                                                                                                                |
| ---------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `itt`                  | `tau_summary.csv` / `rope_summary.csv`                      | `tau_prob_median`, CI, `prob_tau_pos`, `favoured_direction_label`; floored P/N → `offfloor_movers.csv`                                                                                              |
| `joint`                | `tau_contrast_matrix.csv`                                   | pairwise P(effect_i > effect_j); generalisation contrasts                                                                                                                                           |
| `did`                  | `did_summary.csv`                                           | `tau_t2` is the randomised causal contrast; `arm_gap_t1` checks balance; `arm_gap_t3` and `delta_crossover = tau_t2 - arm_gap_t3` are post-crossover associations; every dose term is observational |
| `gain_factors`         | `treatment_marginal.csv`                                    | `trt_prob_*` (the only causal coefficient)                                                                                                                                                          |
| `level_factors`        | `factor_summary.csv`                                        | only `b_grp_time[1]` (t2) is randomised                                                                                                                                                             |
| `mechanism`            | `mechanism_curve.csv` / `interaction_summary.csv`           | slope direction — association, never causal                                                                                                                                                         |
| `mediation` / `_multi` | `mediation_summary.csv`                                     | `total` / `NDE`(IDE) / `NIE`(IIE) / `proportion_mediated`                                                                                                                                           |
| `aligned`              | `cohort_marginal.csv`                                       | association only (cohort contrast not randomised)                                                                                                                                                   |
| `dose_response`        | `dose_slope_summary.csv`                                    | dose slope (collider → sensitivity)                                                                                                                                                                 |
| `adjusted`             | `predictor_associations.csv`                                | adjusted vs bivariate between-child associations                                                                                                                                                    |
| `lcsm`                 | `coupling_summary.csv`                                      | cross-lagged couplings (associations)                                                                                                                                                               |
| `growth`               | `growth_association_summary.csv`                            | `gamma` = ability→growth-rate (association)                                                                                                                                                         |
| `horseshoe`            | `predictor_ranking.csv`                                     | `p_abs_gt_delta` selection                                                                                                                                                                          |
| `corr_factor`          | `factor_correlation_summary.csv` / `structural_summary.csv` | correlations robust; structural leg cautious                                                                                                                                                        |
| `historical_growth`    | `posterior_growth_summary.csv`                              | per-group growth (separate historical study)                                                                                                                                                        |

Record: config, N fitted / failed, gate pass count + divergence caveats, the key randomised ITT τ and `tau_t2` contrasts, `arm_gap_t1` and the post-crossover `arm_gap_t3`/`delta_crossover` quantities separately, the comparison artefacts, and the publish `run_id` + report root. Never collapse the waitlist-crossover quantities into one “DiD treatment effect”. Use the **canonical DAG single-letter outcome codes** (`OUTCOME_LABELS` in `statistical_models/definitions.py`) consistently — do not invent labels.

---

## Step 7 — Pre-commit checks and PR

All three must pass before committing (see CLAUDE.md; do not bypass with `--no-verify`):

```bash
ruff check src/          # Python lint
npm run format:check     # Markdown formatting (Prettier, proseWrap preserve)
npm run spellcheck       # Markdown + Quarto spelling, British English (en-GB)
```

- New markdown must be `git add`-ed before `format:check` / `spellcheck` — those only see tracked files, so an untracked note passes locally then fails CI.
- If cspell flags a legitimate term (identifier, package, domain term, British spelling), add it to `config/spellcheck/allow-en.txt` — do not reword. Only add genuinely-correct terms.

Commit with a Conventional Commit message, open the PR, and reference the issue it closes. The published report URLs and the dated note are the deliverables.

---

## Troubleshooting quick reference

| Symptom                                     | Cause                                                | Fix                                                                             |
| ------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------- |
| Completed fits have no `index.html`         | `--render` batches after all fits; sweep interrupted | Render standalone per dir (Step 3)                                              |
| Sweep died partway                          | external kill / OOM on a slow model                  | Resume only the dirs missing `diagnostics_summary.json` (Step 1), don't restart |
| `--upload` fails with 403                   | VM managed identity lacks the write role             | Use the `AzureCliCredential` wrapper with `AZURE_CLIENT_ID` unset (Step 5)      |
| Script exits 1 but "N fitted, 0 failed"     | the upload step raised, not the fits                 | Read the log; the fits may be fine                                              |
| `xargs` render: "command line … too long"   | BSD xargs + multi-line `sh -c` template              | Use the `render_all.sh` bash loop (Step 3)                                      |
| `quarto render` can't find Python           | `QUARTO_PYTHON` unset                                | `export QUARTO_PYTHON="$(python -c 'import sys;print(sys.executable)')"`        |
| Published dir has no report page            | uploaded before rendering                            | Verify HTML coverage (Step 3) before Step 5                                     |
| A model won't clear the divergence gate     | hard posterior geometry                              | Raise `--target-accept` (0.97 / 0.99); if a funnel, reparameterise              |
| `conda: command not found` in a fresh shell | no env active                                        | `conda activate dse-language-reading-predictors` (Step 0)                       |
