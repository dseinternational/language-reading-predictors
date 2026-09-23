# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Record review coverage without equating a file scan with statistical review.

Drafted by Codex/GPT-6. Reads tracked files only; writes the adjacent JSON ledger.
"""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SM = "src/language_reading_predictors/statistical_models/"
ML = "src/language_reading_predictors/models/"

# These labels describe the coverage achieved, not an approval or clean bill.
COMPLETE_READS = {
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "package.json",
    "src/language_reading_predictors/stats_utils.py",
    ML + "objective.py",
    ML + "lgbm_pipeline.py",
    ML + "lgbm_log_pipeline.py",
    ML + "lgbm_signed_log_pipeline.py",
    ML + "permutation.py",
    ML + "registry.py",
    SM + "posteriors.py",
    SM + "summaries/rope.py",
    SM + "summaries/gain_factors.py",
    SM + "convergence.py",
    SM + "likelihood.py",
    SM + "mediation_integration.py",
    SM + "sampling_quality.py",
    SM + "registry.py",
    SM + "itt_missingness.py",
    SM + "artifacts.py",
    SM + "subfits.py",
    SM + "summaries/dependence.py",
    SM + "release/__init__.py",
    SM + "release/base.py",
    SM + "release/publication.py",
    SM + "release/family_checks.py",
    SM + "release/robustness.py",
    SM + "release/dependence.py",
    SM + "release/blending.py",
    "tests/test_objective.py",
    "tests/statistical_models/test_itt_missingness.py",
    "scripts/assessment_interval_check.py",
    "notes/20260916-codebase-review-and-run-integrity.md",
    "notes/20260913-statistical-review-fixes.md",
    "notes/20260905-statistical-model-review-fixes.md",
    SM + "factories/base.py",
    SM + "factories/itt.py",
    SM + "factories/joint.py",
    SM + "factories/gain_factors.py",
    "docs/models/_partials/_new_child_validation.qmd",
    "docs/models/_partials/_results_joint.qmd",
    "data/readme.md",
    "notes/202607131900-attrition-audit.md",
    "notes/202608051500-decision-key-findings-robustness-release-gate.md",
    "notes/202608201205-itt-code-review-findings.md",
    "notes/202608241500-joint-588-review.md",
    "notes/202608242000-blending-guessing-floor-scope-608.md",
    "notes/202608252100-blending-pair-binding-608-decision-2.md",
    "notes/202609071500-incorporate-deposited-trial-archive.md",
    "notes/202609212100-assessment-interval-lengths.md",
    "notes/202609221800-gb-huber-retune-refit.md",
    "docs/report/chapters/methods-data.qmd",
    "docs/report/chapters/results-itt.qmd",
    "docs/report/_caveats-causal.qmd",
    "docs/report/_caveats-association.qmd",
    "dag/README.md",
    "dag/dag-language-reading.dagitty",
    "dag/dag-reading-language-memory.dagitty",
    "dag/dag-reading-language-memory-lagged.dagitty",
}
SECTION_READS = {
    "dag/dag-language-reading-lagged.dagitty": "All declared edges and most header rationale; timing and reverse-edge interpretation.",
    "docs/report/chapters/methods-models.qmd": "ITT target and selection, equations, marginalisation, computation, prediction and interpretation; selected teaching prose.",
    "METHODS.md": "Boosting, Bayesian design, priors, missingness, computation, prediction, reporting, causal limits, assessment intervals and glossary.",
    "docs/models/README.md": "Inventory headings, ITT, mediation and joint-mechanism descriptions.",
    "src/language_reading_predictors/data_utils.py": "Quarantine, data loading, baseline broadcasting and intervention coding.",
    ML
    + "base_pipeline.py": "Preparation, cross-validation, permutation, construct importance and bootstrap stability.",
    ML + "base_model.py": "Predictor construction, model defaults and automatic registration.",
    ML + "cluster_ranking.py": "Same-skill annotations, linkage choice and aggregate cluster importance.",
    "scripts/tune_model.py": "Search objective, grouped outer and inner splits, target transforms and Huber threshold derivation.",
    SM
    + "new_child_predictive.py": "Plan, reliability checks, latent integration, half-split calculation, PIT and result persistence.",
    SM
    + "new_child_kfold.py": "Child masking, fold groups, parameter transplantation, scoring and completion criteria.",
    SM + "summaries/itt.py": "Per-draw marginal contrast and headline direction, first 240 lines.",
    SM + "summaries/level_factors.py": "Arm-free operating points, focal contrast, link and stored comparator reader.",
    SM + "summaries/did.py": "Declared targets and wave-specific contrast construction.",
    SM + "pipelines/itt.py": "Graded and floor-rule ROPE callers.",
    SM
    + "pipelines/mediation.py": "Persisted family metadata, estimand and outcome-time fields used by release checks.",
    SM
    + "diagnostics.py": "Subfit convergence calculation, lines 915-993, including unassessable and structural parameters.",
    SM + "itt.py": "Settings and moderation validation searches.",
    SM + "factories/horseshoe.py": "Coefficient prior construction, level/gain predictors and RLM builder.",
    SM + "factories/historical.py": "Historical growth likelihood, centring and dispersion construction.",
    SM + "factories/concurrent.py": "Single-wave restrictions, predictor scaling, imputation and group nuisance.",
    SM
    + "factories/joint_mechanism.py": "Conditional slope derivation, covariance interpretation and row restrictions.",
    SM + "factories/lcsm.py": "Change equations, timing assumptions and coupling validation.",
    SM + "factories/growth.py": "Growth equations, age scales, masks and outcome-based intercept anchors.",
    SM + "survival.py": "Person-period construction, missing follow-up, baseline scaling and hazard factory.",
    SM + "mediation.py": "Effect and ratio summaries, single-mediator and period-stacked counterfactual calculations.",
    SM + "factories/mediation.py": "Payload, common baseline vector and shared outcome leg, lines 1-245.",
    SM
    + "preprocessing.py": "Full load_and_prepare function, lines 487-916, covering row selection, covariate timing, count validation and scaling.",
    SM + "family_registry.py": "Family descriptor and resolver dispatch.",
    SM + "stages.py": "Primary fit plan and sampling/PPC sequencing.",
    SM + "definitions.py": "Metadata contract, outcomes and wave contrast semantics.",
    SM + "findings/joint.py": "Contrast-first findings and outcome-scale summaries.",
    "docs/models/_partials/_setup.qmd": "Scientific-output gating and diagnostic visibility.",
    "docs/models/lrp-rli-gbg-012/index.qmd": "Overview, interpretation and findings entry point.",
    "tests/test_permutation_importance.py": "Signal fixture, donor contract and deterministic repeats.",
    "tests/statistical_models/test_new_child_predictive.py": "Reliability and half-split test search.",
    "tests/statistical_models/test_release_decision.py": "Joint contrast width and covariance fixtures, lines 2038-2135; sensitivity and missingness test discovery. Full module executed, not fully read.",
    "tests/statistical_models/test_joint_run_plan.py": "Dependence prior/posterior spread tests, lines 441-541. Full module executed, not fully read.",
    "notes/202609221700-gb-objective-sensitivity.md": "Objective comparisons, interpretation, decision and limitations.",
}
PATTERNS = {
    "huber_mean_claim": r"robust conditional mean|both target the conditional mean|mean-targeting",
    "timing_bound_claim": r"timing bound|first-order bound|interval.*absorbed|length.*absorb",
    "mean_imputation_attenuation_claim": r"imput\w*[^\n.]{0,180}(?:toward|bias|shrink)",
    "causal_identification_language": r"identif(?:y|ies|ied|ication)|causal",
    "cross_validation_language": r"GroupKFold|leave.one|PSIS|cross.validation|new.child",
}


def main() -> None:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    entries = []
    for name in sorted(filter(None, tracked)):
        path = ROOT / name
        raw = path.read_bytes()
        record = {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            record.update(status="binary_inventoried", pending="Content and statistical interpretation not reviewed.")
            entries.append(record)
            continue
        record["lines"] = len(text.splitlines())
        record["status"] = "automated_screen_only"
        record["pending"] = "Detailed statistical and methodological review remains pending."
        record["screen_matches"] = {
            label: [i for i, line in enumerate(text.splitlines(), 1) if re.search(pattern, line, re.I)]
            for label, pattern in PATTERNS.items()
            if re.search(pattern, text, re.I)
        }
        if path.suffix == ".py":
            try:
                ast.parse(text)
                record["python_syntax"] = "parsed"
            except SyntaxError as exc:
                record["python_syntax"] = str(exc)
        if path.suffix == ".ipynb":
            try:
                json.loads(text)
                record["notebook_json"] = "parsed; cell methodology not reviewed"
            except ValueError as exc:
                record["notebook_json"] = str(exc)
        if name in COMPLETE_READS:
            record.update(
                status="full_text_read",
                pending="Read against the questions in this report; not a proof of correctness.",
            )
        elif name in SECTION_READS:
            record.update(status="selected_sections_reviewed", sections=SECTION_READS[name])
        if name.startswith(SM) and re.fullmatch(r"lrp_(rli|rlm)_.*\.py", path.name):
            record["registry_check"] = (
                "Imported and run plan resolved by companion probes; equations not individually approved."
            )
        entries.append(record)
    result = {
        "authorship": "Drafted by Codex/GPT-6",
        "review_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "scope": "All tracked files inventoried; each decodable file screened. Screening and plan resolution are not full statistical review.",
        "coverage_counts": dict(Counter(row["status"] for row in entries)),
        "directory_counts": dict(Counter(row["path"].split("/")[0] for row in entries)),
        "files": entries,
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
