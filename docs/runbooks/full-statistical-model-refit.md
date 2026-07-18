<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).
>
> Substantially edited in the waitlist-crossover, ITT direction-field and release-run workflow guidance by a LLM-based AI tool (Codex/GPT-5).

# Runbook — full statistical-model refit, render, publish and record

A start-to-finish checklist for refitting **every registered Bayesian statistical model** (the Layer-2 PyMC families) at reporting quality, rendering the reports, publishing them to the public research site, and recording the run. Follow it top to bottom; each step says how to check it worked before moving on.

This runbook is the operational companion to the [`lrp-fit-statistical`](../../.claude/skills/lrp-fit-statistical/SKILL.md) skill (which covers the _science_ — estimands, the convergence gate, how to read each family) and to `METHODS.md`. It captures the **workflow and the gotchas** — the things that are not obvious from the scripts and that cost time the first time round. A worked example of its output is `notes/202607131300-full-statistical-refit-reporting.md`.

**Time budget:** the full `reporting` sweep auto-discovers every registered RLI and historical-cohort model and should be treated as a several-hour background job on 16 cores. Most fits are fast (a typical ITT fit is ~40 s under `nutpie`); the slow tail is the growth / mediation / HSGP / LCSM / factor models. Recompute the registry count before each sweep rather than relying on a prose snapshot.

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

Every new publication run must use a **fresh, versioned output root**. Do not refit into bare `output/` or reuse a previous run directory: the fit pipeline writes in place, so old diagnostics, sensitivity files or HTML could otherwise be mistaken for artefacts from the new run. Start from a committed, clean checkout, choose an optional run base, create a unique child directory, and resolve the paths once for every later command:

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
test -z "$(git -C "$REPO_ROOT" status --porcelain)" || {
  echo "Commit or remove worktree changes before a publication run" >&2
  exit 1
}

export RUN_STARTED_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
export GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
export RUN_NAME="${RUN_STARTED_UTC}-$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD)"
export DSE_LRP_RUN_BASE="${DSE_LRP_RUN_BASE:-$REPO_ROOT/output/runs}"
# On a VM, set DSE_LRP_RUN_BASE=/mnt/scratch/lrp-runs before this block.
export DSE_LRP_OUTPUT_DIR="$DSE_LRP_RUN_BASE/$RUN_NAME"
test ! -e "$DSE_LRP_OUTPUT_DIR" || {
  echo "Refusing to reuse existing run root: $DSE_LRP_OUTPUT_DIR" >&2
  exit 1
}
mkdir -p "$DSE_LRP_OUTPUT_DIR"

export OUTPUT_ROOT="$(python -c 'from language_reading_predictors.paths import output_root; print(output_root())')"
export PACKAGE_ROOT="$(python -c 'from language_reading_predictors.paths import ROOT_DIR; print(ROOT_DIR.resolve())')"
test "$PACKAGE_ROOT" = "$(cd "$REPO_ROOT" && pwd -P)" || {
  echo "The active editable install points at $PACKAGE_ROOT, not $REPO_ROOT" >&2
  exit 1
}
export STAT_ROOT="$OUTPUT_ROOT/statistical_models"
export STAT_MODELS_DIR="$STAT_ROOT/models"
export COMPARISON_DIR="$STAT_ROOT/comparison"
export RUN_METADATA_DIR="$OUTPUT_ROOT/run_metadata"
export REPORT_INPUT_SUFFIXES=".qmd .md .json .csv .txt .nc .png .svg .jpg .jpeg .gif .webp .bib .yaml .yml"
mkdir -p "$RUN_METADATA_DIR"
```

Record the immutable inputs before fitting. This manifest captures the exact commit, SHA-256 hash of every file under `data/`, and the shared report-input suffix list used by both freshness checks; it is uploaded with the reports in Step 5:

```bash
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

from language_reading_predictors.paths import DATA_DIR, output_root

data_hashes = {}
for path in sorted(DATA_DIR.rglob("*")):
    if path.is_file():
        with path.open("rb") as source:
            data_hashes[path.relative_to(DATA_DIR).as_posix()] = hashlib.file_digest(
                source, "sha256"
            ).hexdigest()

manifest = {
    "run_name": os.environ["RUN_NAME"],
    "started_utc": os.environ["RUN_STARTED_UTC"],
    "git_sha": os.environ["GIT_SHA"],
    "config": "reporting",
    "output_root": str(output_root()),
    "data_sha256": data_hashes,
    "report_input_suffixes": os.environ["REPORT_INPUT_SUFFIXES"].split(),
}
destination = Path(os.environ["RUN_METADATA_DIR"]) / "run_manifest.json"
destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(destination)
PY
```

The resolved output root is printed at the start of each long-running command and recorded in every model's `config.json`. Keep all exported variables, including `REPORT_INPUT_SUFFIXES`, for the whole workflow. Scratch disks are ephemeral, so complete Step 5 before teardown.

---

## Step 1 — Fit every model (`reporting`)

```bash
python scripts/fit_statistical_model.py all --config reporting --render
```

- `all` fits every registered Layer-2 model sequentially; `--config reporting` = 6 chains × 6000 draws × 6000 tune, `target_accept = 0.95`. `--render` renders each model's Quarto report **in a batch after all fits finish** (see the gotcha below).
- Run it **in the background** with an output log and a persisted exit code so you can watch progress and survive a disconnect:

```bash
nohup bash -c '
  python scripts/fit_statistical_model.py all --config reporting --render
  rc=$?
  printf "%s\n" "$rc" > "$RUN_METADATA_DIR/refit.exit"
  exit "$rc"
' > "$RUN_METADATA_DIR/refit.log" 2>&1 &
echo "$!" > "$RUN_METADATA_DIR/refit.pid"
tail -f "$RUN_METADATA_DIR/refit.log"
```

- **Check `refit.exit`, not just the "N fitted, 0 failed" line.** The script exits non-zero if any fit or render failed, or if the built-in `--upload` step raised. An upload 403 (see Step 5) makes it exit 1 even when every fit and render succeeded, so inspect the log as well as the persisted status.

> [!IMPORTANT]
> **Gotcha — `--render` is batched, not per-model.** `fit_statistical_model.py all --render` runs its Quarto render loop **once, at the very end**, after every model has fitted, iterating an in-memory list built during that process's fit loop. Because this workflow requires a fresh run root, an interrupted sweep leaves completed model directories with `trace.nc`, CSVs and `config.json` but no current `index.html`. See Step 3 for how to render already-fitted directories without re-fitting.

### If the sweep was interrupted: resume, don't restart

Resume only **within the exact same fresh run root and manifest**. In a new shell, re-export the recorded `DSE_LRP_OUTPUT_DIR`, resolve `OUTPUT_ROOT`, `STAT_ROOT`, `STAT_MODELS_DIR`, `COMPARISON_DIR` and `RUN_METADATA_DIR`, and restore `REPORT_INPUT_SUFFIXES` exactly as in Step 0. Confirm that `run_manifest.json` names the current commit, data hashes and suffix list. Do not generate a new `RUN_NAME`, point at an older publication run, or combine artefacts across roots to save time.

Validate that identity before resuming:

```bash
python - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

from language_reading_predictors.paths import DATA_DIR

manifest_path = Path(os.environ["RUN_METADATA_DIR"]) / "run_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
current_sha = subprocess.check_output(
    ["git", "-C", os.environ["REPO_ROOT"], "rev-parse", "HEAD"], text=True
).strip()
current_data = {}
for path in sorted(DATA_DIR.rglob("*")):
    if path.is_file():
        with path.open("rb") as source:
            current_data[path.relative_to(DATA_DIR).as_posix()] = hashlib.file_digest(
                source, "sha256"
            ).hexdigest()
clean = not subprocess.check_output(
    ["git", "-C", os.environ["REPO_ROOT"], "status", "--porcelain"], text=True
).strip()
valid = (
    clean
    and manifest["git_sha"] == current_sha
    and manifest["data_sha256"] == current_data
    and manifest["report_input_suffixes"]
    == os.environ["REPORT_INPUT_SUFFIXES"].split()
    and Path(manifest["output_root"]).resolve() == Path(os.environ["OUTPUT_ROOT"]).resolve()
)
if not valid:
    raise SystemExit(
        "Current checkout, data, report-input suffixes or output root do not match "
        "the run manifest"
    )
print(f"Validated {manifest_path}")
PY
```

After that check, do **not** re-run `all` from scratch — completed fits in this run are intact and expensive. Re-fit only the models in the current manifest's root that lack the sampling artefacts (`diagnostics_summary.json` and `trace.nc`) or finalisation artefacts (`config.json` and the copied `index.qmd`). Requiring all four prevents a pipeline failure after sampling from being mistaken for a completed fit:

```bash
python - <<'PY'
import os
from pathlib import Path

from language_reading_predictors.statistical_models.registry import discover_models

models_dir = Path(os.environ["STAT_MODELS_DIR"])
for model_id in discover_models():
    model_dir = models_dir / f"{model_id}-reporting"
    missing = [
        name
        for name in ("diagnostics_summary.json", "trace.nc", "config.json", "index.qmd")
        if not (model_dir / name).is_file()
    ]
    if missing:
        print(f"MISSING {model_id}: {', '.join(missing)}")
PY
```

Then fit just those, by id:

```bash
python scripts/fit_statistical_model.py lrp-rli-mm-001 --config reporting --render
```

(In the worked example, the first sweep was killed at 112/115; only `lrp-rli-mm-001`, `lrp-rli-mm-101`, `lrp-rlm-hg-001` were re-fitted.)

---

## Step 2 — Verify the fits and triage the convergence gate

**Confirm all models fitted.** Compare the auto-discovered registry count with the number of completed reporting directories; they must match. The following check derives the expected set from the same discovery function as the fit CLI, so no handwritten total is authoritative:

```bash
python - <<'PY'
import os
from pathlib import Path

from language_reading_predictors.statistical_models.registry import discover_models

models_dir = Path(os.environ["STAT_MODELS_DIR"])
registered = set(discover_models())
observed = {path.name.removesuffix("-reporting") for path in models_dir.glob("*-reporting")}
incomplete = {}
for model_id in sorted(registered):
    model_dir = models_dir / f"{model_id}-reporting"
    missing = [
        name
        for name in ("diagnostics_summary.json", "trace.nc", "config.json", "index.qmd")
        if not (model_dir / name).is_file()
    ]
    if missing:
        incomplete[model_id] = missing
print(f"expected={len(registered)} observed={len(observed)}")
for model_id, missing in incomplete.items():
    print(f"INCOMPLETE {model_id}: {', '.join(missing)}")
unexpected = sorted(observed - registered)
if unexpected:
    print(f"UNEXPECTED: {', '.join(unexpected)}")
if observed != registered or incomplete:
    raise SystemExit("Reporting outputs do not exactly match the completed registry")
PY
```

**Read the gate.** Each `diagnostics_summary.json` has `passed` + `checks` = {rhat, ess, divergences, bfmi}. Thresholds: **r̂ ≤ 1.01, ESS ≥ 400, BFMI ≥ 0.30, zero divergences** (the divergence check fails on _any_ divergence). A quick sweep of what passed and what to look at:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

models = Path(os.environ["STAT_MODELS_DIR"])
failed = False
for d in sorted(models.glob("*-reporting")):
    try:
        with (d / "diagnostics_summary.json").open() as fh:
            s = json.load(fh)
    except FileNotFoundError:
        print(f"NO GATE {d.name}")
        failed = True
        continue
    if not s["passed"]:
        print(f"FLAG {d.name}: div={s['divergences']} "
              f"rhat={s['max_rhat']:.4f} ess={s['min_ess']:.0f}")
        failed = True
if failed:
    raise SystemExit("One or more reporting fits did not pass the convergence gate")
PY
```

**Triage — a failed gate is not reporting-ready.** Do not interpret or publish a flagged fit as though it passed, including a fit with only a small number of divergences. Refit after addressing the failure: `--target-accept 0.97` (or `0.99`) is the first lever for isolated divergences, while persistent funnels require reparameterisation. Record failed attempts and the corrective action rather than silently dropping them.

**Sampling convergence is not the whole release gate.** Inspect `pareto_k.csv` and the quantitative predictive tables after every fit. Any `loo_reliable = false` row makes that model's PSIS-LOO score unreliable; do not use its elpd for comparison. In the single-period ITT and joint families, one point is one child, so run a direct leave-out treatment-effect refit excluding all flagged children before claiming robustness. This checks effect stability; it does not recompute an exact leave-one-out elpd. In repeated-measures random-intercept families, one point is a child × phase/period row and the score is conditional on that child's fitted intercept: use an observation-level sensitivity or exact/moment-matched LOO for that same predictive target, and use a separately designed grouped child-level analysis for new-child prediction. Do not mislabel the conditional row-level diagnostic as leave-one-child-out. For joint fits, inspect `posterior_predictive_shape_calibration.csv` as well as the arm/baseline table: an outcome-wide median, upper-quartile or interquartile-range flag qualifies comparisons involving that outcome even if R-hat, ESS and BFMI pass.

For each flagged single-period ITT/joint model, run the checked-in direct leave-out sensitivity command. It verifies the saved specification, data checksum and point-to-child mapping; excludes all children above the saved ArviZ threshold; reuses the completed fit's sampling settings; gates every free variable; and writes the trace plus a provenance-rich comparison centrally and beside the report. The 2026-07-15 reporting audit used:

```bash
python scripts/influence_sensitivity.py lrp-rli-itt-012 lrp-rli-itt-013 lrp-rli-itt-023 --config reporting
```

Do not run this command for a stacked random-intercept family: its high-k point is a conditional child-by-period row, not a whole child. The runner rejects that mismatch.

Run both reporting-quality prior sweeps before interpretation. Keep their archives separate: the first command writes the 44-cell standard treatment-prior, own-baseline-prior, unadjusted-arm and concentration-prior sweep; the second writes the 12-cell estimand-matched P/N floor grid (Bernoulli off-floor outcome among observed baseline-floor children; treatment-prior SD 0.5/1.0/1.5 × age off/on), persists all 12 floor traces and copies each outcome's six-cell CSV beside its report. A P/N report with a power-scaling conflict remains blocked if its grid is absent, incomplete, non-converged or missing traces:

```bash
python scripts/tau_prior_sensitivity.py --config reporting --outcomes R E UR UE T F L W
python scripts/tau_prior_sensitivity.py --config reporting --outcomes P N
```

The floor-sensitivity command retains failed candidates only in its central sensitivity directory, exits non-zero, and does not replace either report-local grid when validation fails. Independently re-run both fail-closed evaluators after the commands succeed. The standard evaluator requires its exact 44-cell outcome/axis contract and reporting configuration, all 44 content-addressed traces, convergence recomputed from those traces, and alignment to the current eight primary ITT fits. The floor evaluator similarly requires both installed grids to be primary-aligned and trace-backed:

```bash
python - <<'PY'
import os
from pathlib import Path

import pandas as pd

from language_reading_predictors.statistical_models.sensitivity import (
    FLOOR_SENSITIVITY_FILENAME,
    FLOOR_SENSITIVITY_MODEL_IDS,
    STANDARD_SENSITIVITY_FILENAME,
    STANDARD_SENSITIVITY_OUTCOMES,
    evaluate_floor_sensitivity,
    evaluate_standard_sensitivity,
    load_primary_floor_reference,
    load_primary_standard_references,
)

stat_dir = Path(os.environ["STAT_ROOT"])
models_dir = Path(os.environ["STAT_MODELS_DIR"])
failed = False
standard_path = stat_dir / "tau_prior_sensitivity" / STANDARD_SENSITIVITY_FILENAME
standard = pd.read_csv(standard_path) if standard_path.is_file() else None
standard_references = load_primary_standard_references(
    models_dir,
    config_name="reporting",
)
standard_status = evaluate_standard_sensitivity(
    standard,
    config_name="reporting",
    requested_outcomes=STANDARD_SENSITIVITY_OUTCOMES,
    primary_references=standard_references,
    trace_root=standard_path.parent,
)
print(f"standard prior sensitivity: {standard_status}")
failed = failed or not standard_status["ready"]

for symbol, model_id in FLOOR_SENSITIVITY_MODEL_IDS.items():
    model_dir = models_dir / f"{model_id}-reporting"
    csv_path = model_dir / FLOOR_SENSITIVITY_FILENAME
    grid = pd.read_csv(csv_path) if csv_path.is_file() else None
    reference = load_primary_floor_reference(
        model_dir,
        symbol,
        config_name="reporting",
    )
    status = evaluate_floor_sensitivity(
        grid,
        symbol,
        primary_reference=reference,
        trace_root=model_dir,
    )
    print(f"{model_id}: {status}")
    failed = failed or not status["ready"]
if failed:
    raise SystemExit("Standard or P/N floor-sensitivity release gate failed")
PY
```

Treat the resulting range as evidence, not merely a checkbox. A complete grid clears the computational gate; material movement in the risk difference or `P(benefit)` is prior sensitivity that must be reported.

> [!IMPORTANT]
> Step 2 creates and copies the P/N and influential-child sensitivity artefacts **after** Step 1's batch render. The affected HTML is therefore stale even when Step 1 completed successfully. Re-render the changed reports in Step 3 before verification or publication.

---

## Step 3 — Refresh changed reports and render any gaps

Render every missing or stale report with the script below. The shared `REPORT_INPUT_SUFFIXES` list from Step 0 defines report inputs recursively: the main QMD and copied partials; Markdown, BibTeX and YAML sources; JSON metadata; CSV and text results; NetCDF traces read by the report validators; and linked raster/vector figures. Quarto's generated `index.html`, `index_files/` resources and render logs are deliberately outside that list. A report is fresh only when `index.html` is newer than **every** current matching input. The renderer deletes the previous HTML before starting, deletes partial output after an error, propagates every Quarto failure, and rechecks the complete input set before recording success. It therefore cannot count an old page as a successful render.

Save this as `/tmp/render_all.sh` and run it (a four-way Bash loop; **do not** use `xargs -I{}` with a multi-line `sh -c` template — BSD xargs dies with "command line cannot be assembled, too long"):

```bash
#!/bin/bash
set -o pipefail
: "${STAT_MODELS_DIR:?Run Step 0 and export STAT_MODELS_DIR first}"
: "${RUN_METADATA_DIR:?Run Step 0 and export RUN_METADATA_DIR first}"
: "${REPORT_INPUT_SUFFIXES:?Run Step 0 and export REPORT_INPUT_SUFFIXES first}"
cd "${REPO_ROOT:?Run Step 0 and export REPO_ROOT first}" || exit 1
export PATH="/Applications/quarto/bin:$PATH"
export QUARTO_PYTHON="$(python -c 'import sys; print(sys.executable)')"
RLOG="$RUN_METADATA_DIR/render_all.log"

report_inputs() {
  local d="$1" source suffix
  find "$d" -type f ! -path "$d/index_files/*" -print |
    while IFS= read -r source; do
      suffix=".${source##*.}"
      case " $REPORT_INPUT_SUFFIXES " in
        *" $suffix "*) printf '%s\n' "$source" ;;
      esac
    done
}

has_required_inputs() {
  local d="$1" source partial_found=0
  for source in \
    "$d/index.qmd" \
    "$d/config.json" \
    "$d/diagnostics_summary.json" \
    "$d/trace.nc"; do
    [ -f "$source" ] || return 1
  done
  for source in "$d"/_partials/*.qmd; do
    if [ -f "$source" ]; then
      partial_found=1
      break
    fi
  done
  [ "$partial_found" -eq 1 ]
}

is_fresh() {
  local d="$1" html="$1/index.html" source
  has_required_inputs "$d" || return 1
  report_inputs "$d" >/dev/null || return 1
  [ -f "$html" ] || return 1
  while IFS= read -r source; do
    [ "$html" -nt "$source" ] || return 1
  done < <(report_inputs "$d")
}

check_all() {
  local d html source found=0 stale=0
  for d in "$STAT_MODELS_DIR"/*-reporting; do
    [ -d "$d" ] || continue
    found=1
    html="$d/index.html"
    if ! has_required_inputs "$d"; then
      echo "MISSING REQUIRED REPORT INPUT: $(basename "$d")"
      stale=1
      continue
    fi
    if ! report_inputs "$d" >/dev/null; then
      echo "REPORT INPUT SCAN FAILED: $(basename "$d")"
      stale=1
      continue
    fi
    if [ ! -f "$html" ]; then
      echo "NO HTML: $(basename "$d")"
      stale=1
      continue
    fi
    while IFS= read -r source; do
      if [ ! "$html" -nt "$source" ]; then
        echo "STALE HTML: $(basename "$d") older than ${source#"$d"/}"
        stale=1
      fi
    done < <(report_inputs "$d")
  done
  if [ "$found" -eq 0 ]; then
    echo "NO REPORTING DIRECTORIES under $STAT_MODELS_DIR"
    return 1
  fi
  [ "$stale" -eq 0 ]
}

if [ "${1:-}" = "--check-only" ]; then
  check_all
  exit $?
fi

: > "$RLOG"

render_one() {
  local d="$1" html="$1/index.html"
  rm -f "$html"
  if ! (cd "$d" && quarto render index.qmd >render.log 2>render.err); then
    rm -f "$html"
    echo "FAIL $(basename "$d"): Quarto error" >>"$RLOG"
    return 1
  fi
  if ! is_fresh "$d"; then
    rm -f "$html"
    echo "FAIL $(basename "$d"): missing or stale HTML" >>"$RLOG"
    return 1
  fi
  echo "OK $(basename "$d")" >>"$RLOG"
}

MAXJOBS=4
pids=()
failed=0
for d in "$STAT_MODELS_DIR"/*-reporting; do
  [ -d "$d" ] || continue
  is_fresh "$d" && continue
  render_one "$d" &
  pids+=("$!")
  if [ "${#pids[@]}" -eq "$MAXJOBS" ]; then
    for pid in "${pids[@]}"; do
      wait "$pid" || failed=1
    done
    pids=()
  fi
done
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if ! check_all >>"$RLOG"; then
  failed=1
fi
echo "DONE OK=$(grep -c '^OK' "$RLOG") FAIL=$(grep -c '^FAIL' "$RLOG")" >>"$RLOG"
tail -5 "$RLOG"
exit "$failed"
```

```bash
bash /tmp/render_all.sh
```

**Verify current HTML before uploading.** The check-only mode independently applies the same complete input list and fails for any missing or stale page:

```bash
bash /tmp/render_all.sh --check-only
```

---

## Step 4 — Cross-model comparison

```bash
python scripts/compare_statistical_models.py --config reporting
```

Writes to `$COMPARISON_DIR`: `itt_vs_joint_tau.csv` (single-outcome τ vs joint τ_k consistency), `tau_forest.png`, `mediation_family.csv` + `mediation_family_forest.png`, and nested PSIS-LOO tables (mechanism / phonics-route / age-moderation / dose / did-dose). The script must exclude gate-flagged models from interpretable LOO tables and verify identical observation identities and order before comparing models. Note every artefact it skips and why; equal row counts alone are not sufficient evidence that pointwise LOO values are aligned.

---

## Step 5 — Publish to the public research site

> [!IMPORTANT]
> **Public + preliminary — confirm the scope with the study owner before publishing.** These reports become anonymously readable on the internet.

> [!IMPORTANT]
> **Gotcha — the built-in `--upload` 403s from a VM.** The public `dseresearch` container needs the write role. `az login` (Frank's identity) has it; the VM **managed identity** does not, and `DefaultAzureCredential` prefers the MI — so the built-in flag fails with 403. Publish with the small wrapper below, which forces `AzureCliCredential`, and run it with `AZURE_CLIENT_ID` unset so nothing re-selects the MI. To make `--upload` first-class instead, grant the runner identity **Storage Blob Data Contributor** on the `dseresearch` account.

The wrapper independently re-runs the machine-checkable scientific release gates; it does not assume that Steps 2–3 happened merely because their files exist. It validates the clean checkout, commit, data hashes and output root both before the checks and immediately before upload. It derives every model kind from the current registered specification and requires the saved `config.json` kind to match before using that registered kind to route the ITT/joint influence gate. It refuses any failed convergence summary, a malformed or internally inconsistent ITT/joint Pareto table, any high-k ITT/joint fit without a current converged direct leave-out bundle tied to the same fit, a standard prior-sensitivity archive that is incomplete, trace-invalid or no longer aligned to the eight primary ITT fits, either P/N model without a current complete and trace-validated floor-sensitivity grid, and any HTML page that is missing or older than one of the complete report inputs defined in Step 0. Passing these checks does not replace scientific review of the sensitivity effect sizes or the study owner's publication approval.

Save as `/tmp/upload_public.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from azure.identity import AzureCliCredential
from dse_research_utils.storage.azure import upload_directory_to_blob_storage
from language_reading_predictors.paths import DATA_DIR, output_root
from language_reading_predictors.statistical_models.influence import (
    evaluate_influence_bundle,
)
from language_reading_predictors.statistical_models.registry import discover_models
from language_reading_predictors.statistical_models.sensitivity import (
    FLOOR_SENSITIVITY_FILENAME,
    FLOOR_SENSITIVITY_MODEL_IDS,
    STANDARD_SENSITIVITY_FILENAME,
    STANDARD_SENSITIVITY_OUTCOMES,
    evaluate_floor_sensitivity,
    evaluate_standard_sensitivity,
    load_primary_floor_reference,
    load_primary_standard_references,
    tau_psense_status,
)

credential = AzureCliCredential()
run_id = str(uuid.uuid7())            # ONE run_id for the whole batch
print(f"RUN_ID={run_id}", flush=True)

models_dir = Path(os.environ["STAT_MODELS_DIR"])
comparison_dir = Path(os.environ["COMPARISON_DIR"])
run_metadata_dir = Path(os.environ["RUN_METADATA_DIR"])
standard_sensitivity_dir = Path(os.environ["STAT_ROOT"]) / "tau_prior_sensitivity"
floor_sensitivity_dir = Path(os.environ["STAT_ROOT"]) / "floor_tau_prior_sensitivity"
influence_sensitivity_dir = Path(os.environ["STAT_ROOT"]) / "influence_sensitivity"
report_input_suffixes = tuple(os.environ["REPORT_INPUT_SUFFIXES"].split())
if (
    not report_input_suffixes
    or len(report_input_suffixes) != len(set(report_input_suffixes))
    or any(
        not suffix.startswith(".") or suffix != suffix.lower()
        for suffix in report_input_suffixes
    )
):
    raise SystemExit("REPORT_INPUT_SUFFIXES is empty, duplicated or malformed")
model_targets = sorted(models_dir.glob("*-reporting"))
targets = [
    *model_targets,
    comparison_dir,
    standard_sensitivity_dir,
    floor_sensitivity_dir,
    influence_sensitivity_dir,
    run_metadata_dir,
]
missing = [str(path) for path in targets if not path.is_dir()]
manifest_path = run_metadata_dir / "run_manifest.json"
if not manifest_path.is_file():
    missing.append(str(manifest_path))
if missing:
    raise SystemExit(f"Refusing an incomplete upload; missing: {missing}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_manifest():
    current_sha = subprocess.check_output(
        ["git", "-C", os.environ["REPO_ROOT"], "rev-parse", "HEAD"], text=True
    ).strip()
    clean = not subprocess.check_output(
        ["git", "-C", os.environ["REPO_ROOT"], "status", "--porcelain"], text=True
    ).strip()
    current_data = {}
    for path in sorted(DATA_DIR.rglob("*")):
        if path.is_file():
            with path.open("rb") as source:
                current_data[path.relative_to(DATA_DIR).as_posix()] = hashlib.file_digest(
                    source, "sha256"
                ).hexdigest()
    try:
        manifest_output_root = Path(manifest["output_root"]).resolve()
    except (KeyError, TypeError):
        raise SystemExit("Refusing upload: run manifest has no valid output root") from None
    resolved_output_root = output_root().resolve()
    valid = (
        clean
        and manifest.get("git_sha") == current_sha
        and manifest.get("data_sha256") == current_data
        and manifest.get("config") == "reporting"
        and manifest.get("report_input_suffixes") == list(report_input_suffixes)
        and manifest_output_root == resolved_output_root
        and manifest_output_root == Path(os.environ["OUTPUT_ROOT"]).resolve()
    )
    if not valid:
        raise SystemExit(
            "Refusing upload: current clean checkout, commit, data, reporting config, "
            "report-input suffixes or output root do not match the run manifest"
        )


validate_manifest()

registered_models = discover_models()
registered_kinds = {}
for model_id, module in registered_models.items():
    spec = getattr(module, "SPEC", None)
    if spec is None:
        get_spec = getattr(module, "get_spec", None)
        spec = get_spec() if callable(get_spec) else None
    spec_model_id = getattr(spec, "model_id", None)
    spec_kind = getattr(spec, "kind", None)
    if spec_model_id != model_id or not isinstance(spec_kind, str) or not spec_kind:
        raise SystemExit(f"Refusing upload: no valid registered identity for {model_id}")
    registered_kinds[model_id] = spec_kind

expected = {f"{model_id}-reporting" for model_id in registered_models}
observed = {path.name for path in model_targets}
if observed != expected:
    raise SystemExit("Refusing upload: model directories do not match the registry")

release_errors = []
configs = {}
for model_dir in model_targets:
    diagnostics_path = model_dir / "diagnostics_summary.json"
    config_path = model_dir / "config.json"
    try:
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        release_errors.append(f"{model_dir.name}: unreadable release metadata ({exc})")
        continue
    model_id = model_dir.name.removesuffix("-reporting")
    if diagnostics.get("passed") is not True:
        release_errors.append(f"{model_dir.name}: convergence gate did not pass")
    if config.get("model_id") != model_id:
        release_errors.append(f"{model_dir.name}: config model identity mismatch")
    if config.get("kind") != registered_kinds[model_id]:
        release_errors.append(
            f"{model_dir.name}: saved kind {config.get('kind')!r} does not match "
            f"registered kind {registered_kinds[model_id]!r}"
        )
    configs[model_dir.name] = config

standard_path = standard_sensitivity_dir / STANDARD_SENSITIVITY_FILENAME
standard = pd.read_csv(standard_path) if standard_path.is_file() else None
standard_references = load_primary_standard_references(
    models_dir,
    config_name="reporting",
)
standard_status = evaluate_standard_sensitivity(
    standard,
    config_name="reporting",
    requested_outcomes=STANDARD_SENSITIVITY_OUTCOMES,
    primary_references=standard_references,
    trace_root=standard_sensitivity_dir,
)
if not standard_status["ready"]:
    release_errors.append(
        "standard prior-sensitivity archive is invalid: "
        f"{standard_status}"
    )

for model_dir in model_targets:
    config = configs.get(model_dir.name)
    model_id = model_dir.name.removesuffix("-reporting")
    if config is None or registered_kinds[model_id] not in {"itt", "joint"}:
        continue
    pareto_path = model_dir / "pareto_k.csv"
    if not pareto_path.is_file():
        release_errors.append(f"{model_dir.name}: missing Pareto-k table")
        continue
    pareto = pd.read_csv(pareto_path)
    required_pareto = {
        "observation_index",
        "subject_id",
        "pareto_k",
        "good_k_threshold",
        "loo_reliable",
    }
    if not required_pareto.issubset(pareto.columns) or pareto.empty:
        release_errors.append(f"{model_dir.name}: malformed Pareto-k table")
        continue
    try:
        n_obs = int(config["n_obs"])
    except (KeyError, TypeError, ValueError):
        release_errors.append(f"{model_dir.name}: config has no valid n_obs")
        continue
    observation_indices = pd.to_numeric(
        pareto["observation_index"], errors="coerce"
    )
    pareto_k = pd.to_numeric(pareto["pareto_k"], errors="coerce")
    thresholds = pd.to_numeric(pareto["good_k_threshold"], errors="coerce")
    reliability = pareto["loo_reliable"].map(
        lambda value: (
            bool(value)
            if isinstance(value, (bool, np.bool_))
            else {"true": True, "false": False}.get(str(value).strip().lower())
        )
    )
    numeric = np.column_stack((observation_indices, pareto_k, thresholds))
    if (
        n_obs <= 0
        or len(pareto) != n_obs
        or not np.isfinite(numeric).all()
        or not np.equal(observation_indices, np.floor(observation_indices)).all()
        or observation_indices.astype(int).duplicated().any()
        or set(observation_indices.astype(int)) != set(range(n_obs))
        or pareto["subject_id"].isna().any()
        or pareto["subject_id"].astype(str).str.strip().eq("").any()
        or pareto["subject_id"].astype(str).duplicated().any()
        or thresholds.nunique() != 1
        or reliability.isna().any()
        or not reliability.astype(bool).eq(pareto_k <= thresholds).all()
    ):
        release_errors.append(
            f"{model_dir.name}: Pareto-k rows do not map one-to-one onto all fitted children"
        )
        continue
    threshold = float(thresholds.iloc[0])
    flagged = pareto.loc[pareto_k > threshold]
    if flagged.empty:
        continue

    influence_path = model_dir / "influence_sensitivity.csv"
    influence = pd.read_csv(influence_path) if influence_path.is_file() else None
    influence_status = evaluate_influence_bundle(
        influence,
        model_dir,
        config,
        "reporting",
    )
    if not influence_status["ready"]:
        release_errors.append(
            f"{model_dir.name}: high-k influence bundle is invalid: "
            f"{influence_status['reason']}"
        )

for symbol, model_id in FLOOR_SENSITIVITY_MODEL_IDS.items():
    model_dir = models_dir / f"{model_id}-reporting"
    psense_path = model_dir / "psense_summary.csv"
    psense = pd.read_csv(psense_path, index_col=0) if psense_path.is_file() else None
    psense_status = tau_psense_status(psense)
    floor_path = model_dir / FLOOR_SENSITIVITY_FILENAME
    floor_grid = pd.read_csv(floor_path) if floor_path.is_file() else None
    try:
        reference = load_primary_floor_reference(
            model_dir,
            symbol,
            config_name="reporting",
        )
        floor_status = evaluate_floor_sensitivity(
            floor_grid,
            symbol,
            primary_reference=reference,
            trace_root=model_dir,
        )
    except (OSError, ValueError) as exc:
        release_errors.append(f"{model_dir.name}: floor gate could not run ({exc})")
        continue
    if not floor_status["ready"]:
        release_errors.append(
            f"{model_dir.name}: floor gate is not ready (tau psense={psense_status}; "
            f"status={floor_status})"
        )

if release_errors:
    raise SystemExit("Refusing upload:\n- " + "\n- ".join(release_errors))


def report_inputs(model_dir):
    for path in model_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(model_dir)
        if "index_files" in relative.parts:
            continue
        if path.suffix.lower() in report_input_suffixes:
            yield path


stale = []
for model_dir in model_targets:
    html = model_dir / "index.html"
    required_report_inputs = [
        model_dir / "index.qmd",
        model_dir / "config.json",
        model_dir / "diagnostics_summary.json",
        model_dir / "trace.nc",
    ]
    missing_report_inputs = [
        str(path.relative_to(model_dir))
        for path in required_report_inputs
        if not path.is_file()
    ]
    if not any((model_dir / "_partials").glob("*.qmd")):
        missing_report_inputs.append("_partials/*.qmd")
    if missing_report_inputs:
        stale.append(f"{model_dir.name}: missing {missing_report_inputs}")
        continue
    if not html.is_file():
        stale.append(f"{model_dir.name}: missing index.html")
        continue
    html_mtime = html.stat().st_mtime_ns
    for source in report_inputs(model_dir):
        if html_mtime <= source.stat().st_mtime_ns:
            stale.append(f"{model_dir.name}: {source.relative_to(model_dir)}")
if stale:
    raise SystemExit(f"Refusing upload: missing or stale HTML inputs: {stale}")

# Recompute the checkout and data identity immediately before the first upload.
validate_manifest()

trace_backed_targets = {
    standard_sensitivity_dir,
    floor_sensitivity_dir,
    influence_sensitivity_dir,
}
ok, fail = {}, []
for d in targets:
    name = d.name
    try:
        res = upload_directory_to_blob_storage(
            str(d), name, project="language-reading-predictors",
            # Keep the certified sensitivity evidence; omit the ~16 GB model traces.
            include_traces=d in trace_backed_targets,
            run_id=run_id, credential=credential)
        ok[name] = res.report_url
        print(f"OK {name} -> {res.report_url}", flush=True)
    except Exception as exc:           # noqa: BLE001
        fail.append((name, repr(exc)))
        print(f"FAIL {name}: {exc!r}", flush=True)

print(f"\nUPLOADED={len(ok)} FAILED={len(fail)}  RUN_ID={run_id}")
if fail:
    raise SystemExit(1)
```

The standard, floor and influential-child sensitivity targets upload their content-addressed NetCDF traces because their certified manifests and report-local comparisons point to that evidence. The much larger primary per-model `trace.nc` files remain excluded. All three central sensitivity directories are required targets, so a missing archive stops the release rather than producing unverifiable report claims.

Run it with the MI de-selected:

```bash
unset AZURE_CLIENT_ID
python /tmp/upload_public.py
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
| `itt`                  | `tau_summary.csv` / `rope_summary.csv`                      | `tau_prob_median`, CI, `prob_ame_pos`, `favoured_direction_label` (`prob_tau_pos` is a compatibility alias); floored P/N → `offfloor_movers.csv`                                                    |
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

Record: the local `RUN_NAME`, `run_manifest.json` commit and data hashes, config, N fitted / failed, gate pass count + divergence caveats, the key randomised ITT τ and `tau_t2` contrasts, `arm_gap_t1` and the post-crossover `arm_gap_t3`/`delta_crossover` quantities separately, the comparison artefacts, and the publish `run_id` + report root. Never collapse the waitlist-crossover quantities into one “DiD treatment effect”. Use the **canonical DAG single-letter outcome codes** (`OUTCOME_LABELS` in `statistical_models/definitions.py`) consistently — do not invent labels.

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

| Symptom                                     | Cause                                                | Fix                                                                         |
| ------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------- |
| Completed fits have no `index.html`         | `--render` batches after all fits; sweep interrupted | Run the freshness-aware standalone renderer (Step 3)                        |
| Sweep died partway                          | external kill / OOM on a slow model                  | Resume missing fits only within the same manifested fresh run root (Step 1) |
| `--upload` fails with 403                   | VM managed identity lacks the write role             | Use the `AzureCliCredential` wrapper with `AZURE_CLIENT_ID` unset (Step 5)  |
| Script exits 1 but "N fitted, 0 failed"     | the upload step raised, not the fits                 | Read the log; the fits may be fine                                          |
| `xargs` render: "command line … too long"   | BSD xargs + multi-line `sh -c` template              | Use the `render_all.sh` bash loop (Step 3)                                  |
| `quarto render` can't find Python           | `QUARTO_PYTHON` unset                                | `export QUARTO_PYTHON="$(python -c 'import sys;print(sys.executable)')"`    |
| Published dir has no current report page    | missing, failed or stale render                      | Run the renderer and its check-only mode in Step 3 before Step 5            |
| A model won't clear the divergence gate     | hard posterior geometry                              | Raise `--target-accept` (0.97 / 0.99); if a funnel, reparameterise          |
| `conda: command not found` in a fresh shell | no env active                                        | `conda activate dse-language-reading-predictors` (Step 0)                   |
