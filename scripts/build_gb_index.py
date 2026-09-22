# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build an HTML index page linking every rendered gradient-boosting report.

Each fit renders its own ``<model_id>/index.html`` but nothing links them. This
script writes ``index.html`` beside the fit directories (``output/models/`` by
default). The page pairs each gain model with its level model, grouped as in
the Layer 1 tables of ``docs/models/README.md``. Each model shows its pooled
held-out R², MAE, row count and its three most important predictors, with SHAP
direction and bootstrap stability. Registered or fitted models missing from the
catalogue are listed separately, so the page never hides a fit.

Everything is read from stored artefacts (``metrics.json``, ``config.json`` and
``predictor_ranking.csv``); nothing is refitted. Rerun it after a sweep::

    python scripts/build_gb_index.py
    python scripts/build_gb_index.py --output-dir /scratch/run-root
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from language_reading_predictors import paths  # noqa: E402

N_LEAD_PREDICTORS = 3

_ROW_RE = re.compile(
    r"^\|\s*(`lrp-rli-gbg-\d+`|—)\s*\|\s*(`lrp-rli-gbl-\d+`|—)\s*\|\s*(.+?)\s*\|\s*$"
)


@dataclass(frozen=True)
class CatalogueRow:
    """One measure: its gain model, level model and outcome label."""

    gain: str | None
    level: str | None
    outcome: str


@dataclass
class CatalogueSection:
    title: str
    rows: list[CatalogueRow] = field(default_factory=list)


def parse_catalogue(text: str) -> list[CatalogueSection]:
    """Read the gain/level tables under the catalogue's "Layer 1" heading."""
    sections: list[CatalogueSection] = []
    in_layer1 = False
    for line in text.splitlines():
        if line.startswith("## "):
            in_layer1 = line.startswith("## Layer 1")
            continue
        if not in_layer1:
            continue
        if line.startswith("### "):
            title = re.sub(r"\s*\(`.*$", "", line[4:]).strip()
            sections.append(CatalogueSection(title))
            continue
        match = _ROW_RE.match(line)
        if match and sections:
            gain, level, outcome = match.groups()
            sections[-1].rows.append(
                CatalogueRow(
                    gain=None if gain == "—" else gain.strip("`"),
                    level=None if level == "—" else level.strip("`"),
                    outcome=outcome,
                )
            )
    return sections


def _inline_code(text: str) -> str:
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(text))


def _link(target: Path, page_dir: Path) -> str:
    return Path(os.path.relpath(target, page_dir)).as_posix()


def _lead_predictors(ranking: pd.DataFrame, target: str) -> str:
    """The top predictors by permutation importance, as the report narrates them."""
    ordered = ranking.sort_values("perm_imp_mean", ascending=False)
    positive = ordered[ordered["perm_imp_mean"] > 0]
    lead = (positive if not positive.empty else ordered).head(N_LEAD_PREDICTORS)
    items = []
    for _, row in lead.iterrows():
        sign = str(row.get("sign", "")).strip()
        arrow, css, direction = {
            "+": ("↑", "sp", "higher values go with a larger predicted"),
            "-": ("↓", "sn", "higher values go with a smaller predicted"),
        }.get(sign, ("·", "sz", "no single direction for"))
        same_skill = bool(row.get("same_skill_of_outcome", False))
        topk = row.get("topk_freq")
        has_topk = topk is not None and pd.notna(topk)
        tip = (
            f"{row['member']}: {direction} {target}. Permutation importance "
            f"{row['perm_imp_mean']:.3g} ± {row['perm_imp_sd']:.2g} (held-out RMSE change); "
            f"mean |SHAP| {row['mean_abs_shap']:.3g}"
            + (f"; top five in {topk:.0%} of bootstrap refits" if has_topk else "")
            + ("; same skill as the outcome" if same_skill else "")
        )
        items.append(
            f'<li title="{html.escape(tip)}"><span class="arrow {css}">{arrow}</span>'
            f"<code>{html.escape(str(row['member']))}</code>"
            + ('<span class="flag">same skill</span>' if same_skill else "")
            + (f'<span class="stab">{topk:.0%}</span>' if has_topk else "")
            + "</li>"
        )
    return '<ol class="preds">' + "".join(items) + "</ol>"


def _model_half(models_dir: Path, page_dir: Path, model_id: str | None, kind: str) -> str:
    if model_id is None:
        return (
            f'<div class="half empty"><span class="kind">{kind}</span>'
            f'<p class="muted">No {kind.lower()} model for this measure.</p></div>'
        )
    fit_dir = models_dir / model_id
    if not (fit_dir / "metrics.json").is_file():
        return (
            f'<div class="half empty"><span class="kind">{kind}</span>'
            f'<p class="muted"><code>{html.escape(model_id)}</code> has not been fitted here.</p></div>'
        )
    metrics = json.loads((fit_dir / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((fit_dir / "config.json").read_text(encoding="utf-8"))
    target = str(config.get("target_var", ""))

    report = fit_dir / "index.html"
    name = html.escape(model_id)
    head = (
        f'<a href="{html.escape(_link(report, page_dir))}">{name}</a>'
        if report.is_file()
        else f'{name} <span class="muted">(not rendered)</span>'
    )
    if metrics.get("fit_complete") is not True:
        head += '<span class="warn">completion not recorded</span>'

    def number(key: str, fmt: str) -> tuple[str, float | None]:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return format(value, fmt), float(value)
        return "–", None

    r2_text, r2 = number("cv_pooled_r2", ".2f")
    mae_text, _ = number("cv_pooled_mae", ".2f")
    n_text, _ = number("n_observations", "d")
    r2_class = ' class="neg"' if r2 is not None and r2 < 0 else ""

    ranking_path = fit_dir / "predictor_ranking.csv"
    predictors = (
        _lead_predictors(pd.read_csv(ranking_path), target)
        if ranking_path.is_file()
        else '<p class="muted">No predictor ranking (development fit?).</p>'
    )
    return f"""<div class="half">
<div class="head"><span class="kind">{kind}</span>{head}</div>
<div class="stats"><span title="Pooled held-out R² across child-grouped folds; hyperparameters were tuned on the same folds">R² <b{r2_class}>{r2_text}</b></span>
<span title="Pooled held-out mean absolute error, in outcome units">MAE <b>{mae_text}</b></span>
<span title="Rows used in the fit">n <b>{n_text}</b></span></div>
{predictors}
</div>"""


def _provenance_line(models_dir: Path, fitted: Iterable[str]) -> str:
    commits: set[str] = set()
    dirty: set[object] = set()
    configs: set[str] = set()
    times: list[datetime] = []
    count = 0
    for model_id in fitted:
        count += 1
        config = json.loads((models_dir / model_id / "config.json").read_text(encoding="utf-8"))
        provenance = config.get("provenance") or {}
        source = provenance.get("source") or {}
        commits.add(str(source.get("commit") or "unknown")[:8])
        dirty.add(source.get("dirty"))
        configs.add(str(config.get("run_config")))
        recorded = provenance.get("recorded_at_utc")
        if isinstance(recorded, str):
            times.append(datetime.fromisoformat(recorded).astimezone())
    line = (
        f"{count} fitted models · config <code>{html.escape(', '.join(sorted(configs)))}</code>"
        f" · commit <code>{html.escape(', '.join(sorted(commits)))}</code>"
        + (" · clean source tree" if dirty == {False} else " · <span class=\"warn\">source not recorded clean for every fit</span>")
    )
    if times:
        first, last = min(times), max(times)
        span = (
            f"{first:%d %b %Y %H:%M}–{last:%H:%M}"
            if first.date() == last.date()
            else f"{first:%d %b %Y} – {last:%d %b %Y}"
        )
        line += f" · fitted {span}"
    return line


def build_page(
    models_dir: Path,
    catalogue_text: str,
    model_ids: Iterable[str],
    page_dir: Path,
) -> str:
    """Return the index HTML for the fits under ``models_dir``."""
    sections = parse_catalogue(catalogue_text)
    catalogued = {
        model_id
        for section in sections
        for row in section.rows
        for model_id in (row.gain, row.level)
        if model_id
    }
    on_disk = {
        entry.name
        for entry in models_dir.iterdir()
        if entry.is_dir() and (entry / "metrics.json").is_file()
    } if models_dir.is_dir() else set()
    extra = sorted((set(model_ids) | on_disk) - catalogued)
    if extra:
        sections.append(
            CatalogueSection(
                "Not in the catalogue",
                [
                    CatalogueRow(
                        gain=model_id if "-gbg-" in model_id else None,
                        level=None if "-gbg-" in model_id else model_id,
                        outcome=f"`{model_id}`",
                    )
                    for model_id in extra
                ],
            )
        )

    body = []
    for section in sections:
        cards = "".join(
            f'<article class="card"><h3>{_inline_code(row.outcome)}</h3><div class="pair">'
            f"{_model_half(models_dir, page_dir, row.gain, 'Gain')}"
            f"{_model_half(models_dir, page_dir, row.level, 'Level')}</div></article>"
            for row in section.rows
        )
        body.append(f"<section><h2>{html.escape(section.title)}</h2>{cards}</section>")

    return _PAGE.format(
        provenance=_provenance_line(models_dir, sorted(on_disk)),
        body="".join(body),
        generated=f"{datetime.now():%d %b %Y %H:%M}",
    )


_PAGE = """<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GB model reports</title>
<style>
:root {{
  --bg: #fbfbf9; --panel: #ffffff; --ink: #1d1d1b; --muted: #6b6b66; --line: #e4e3de;
  --accent: #1f5fae; --pos: #1d7a46; --neg: #b3402e; --warn-bg: #fff4d6; --warn: #7a5600;
  --flag-bg: #fde6e2; --flag: #9b2c1c; --code-bg: #f1f0ec;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #161615; --panel: #1f1f1d; --ink: #ecebe6; --muted: #a09f98; --line: #34332f;
    --accent: #7fb0ec; --pos: #6fcf97; --neg: #f08a76; --warn-bg: #3a3014; --warn: #f2cf73;
    --flag-bg: #43231d; --flag: #f3a595; --code-bg: #2a2a27;
  }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 32px 16px 64px; }}
h1 {{ font-size: 1.6rem; margin: 0 0 4px; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.1rem; margin: 40px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }}
h3 {{ font-size: 0.98rem; margin: 0 0 10px; font-weight: 600; }}
code {{ font: 0.86em ui-monospace, SFMono-Regular, Menlo, monospace; background: var(--code-bg); padding: 1px 4px; border-radius: 4px; }}
a {{ color: var(--accent); text-decoration: none; font-weight: 600; }}
a:hover {{ text-decoration: underline; }}
.muted {{ color: var(--muted); }}
.prov {{ color: var(--muted); font-size: 0.88rem; margin: 0 0 16px; }}
.guide {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px; font-size: 0.9rem; }}
.guide p {{ margin: 6px 0; }}
.tools {{ position: sticky; top: 0; background: var(--bg); padding: 12px 0; z-index: 1; }}
.tools input {{ width: 100%; max-width: 420px; padding: 8px 12px; border: 1px solid var(--line); border-radius: 8px;
  background: var(--panel); color: var(--ink); font: inherit; }}
.card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; margin: 0 0 12px; }}
.pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
.half {{ min-width: 0; }}
.half + .half {{ border-left: 1px solid var(--line); padding-left: 16px; }}
@media (max-width: 720px) {{
  .pair {{ grid-template-columns: 1fr; }}
  .half + .half {{ border-left: 0; padding-left: 0; border-top: 1px solid var(--line); padding-top: 12px; }}
}}
.head {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px; }}
.kind {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); font-weight: 600; min-width: 3.2em; }}
.stats {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.86rem; color: var(--muted); margin: 4px 0 6px; }}
.stats b {{ color: var(--ink); font-variant-numeric: tabular-nums; }}
.stats b.neg {{ color: var(--neg); }}
.preds {{ margin: 0; padding: 0; list-style: none; font-size: 0.9rem; }}
.preds li {{ display: flex; align-items: center; gap: 6px; padding: 2px 0; }}
.arrow {{ width: 1.1em; text-align: center; font-weight: 700; }}
.arrow.sp {{ color: var(--pos); }} .arrow.sn {{ color: var(--neg); }} .arrow.sz {{ color: var(--muted); }}
.stab {{ margin-left: auto; color: var(--muted); font-size: 0.8rem; font-variant-numeric: tabular-nums; }}
.flag {{ background: var(--flag-bg); color: var(--flag); font-size: 0.72rem; padding: 0 6px; border-radius: 999px; }}
.warn {{ background: var(--warn-bg); color: var(--warn); font-size: 0.75rem; padding: 0 6px; border-radius: 999px; }}
.empty p {{ margin: 4px 0; font-size: 0.88rem; }}
footer {{ margin-top: 40px; color: var(--muted); font-size: 0.82rem; }}
.hidden {{ display: none; }}
</style>
</head>
<body>
<main>
<h1>Gradient-boosting model reports</h1>
<p class="prov">{provenance}</p>
<div class="guide">
<p><b>Reading this page.</b> Each card pairs the gain model (change to the next assessment) with the level model (concurrent score) for one measure. <b>R²</b> is pooled held-out R² across child-grouped folds. Hyperparameters were tuned on the same folds, so it is internal performance, not independent validation. A negative value (red) means the model predicts worse than the outcome mean.</p>
<p><b>Predictors</b> are the three with the largest permutation importance: the change in held-out RMSE when a predictor is shuffled between children. The arrow is the SHAP direction: <span class="arrow sp">↑</span> higher values go with a larger prediction, <span class="arrow sn">↓</span> with a smaller one, <span class="arrow sz">·</span> no single direction. The percentage is how often the predictor ranked in the top five across 30 child-level bootstrap refits. Hover over a predictor for its numbers. <span class="flag">same skill</span> marks a measure of the outcome's own skill, which can restate a level outcome. Gain models include the baseline score, so a negative baseline association can reflect score limits or regression to the mean. These are predictive associations, not causal effects.</p>
</div>
<div class="tools"><input id="q" type="search" placeholder="Filter by model, outcome or predictor…" aria-label="Filter models"></div>
{body}
<footer>Generated {generated} by <code>scripts/build_gb_index.py</code> from each fit's <code>metrics.json</code>, <code>config.json</code> and <code>predictor_ranking.csv</code>. Groupings follow <code>docs/models/README.md</code>. Reading notes drafted by a LLM-based AI tool (Claude Code/Opus 5).</footer>
</main>
<script>
const q = document.getElementById('q');
q.addEventListener('input', () => {{
  const t = q.value.trim().toLowerCase();
  document.querySelectorAll('.card').forEach(c => {{
    c.classList.toggle('hidden', Boolean(t) && !c.textContent.toLowerCase().includes(t));
  }});
  document.querySelectorAll('section').forEach(s => {{
    s.classList.toggle('hidden', !s.querySelector('.card:not(.hidden)'));
  }});
}});
</script>
</body>
</html>
"""


def _registered_model_ids() -> list[str]:
    from language_reading_predictors.models.registry import MODELS

    return list(MODELS)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=None, help="Override the output root (default: DSE_LRP_OUTPUT_DIR or output/)")
    parser.add_argument("--out", type=Path, default=None, help="Page path (default: <output root>/models/index.html)")
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=paths.DOCS_DIR / "models" / "README.md",
        help="Model catalogue supplying groups and outcome labels",
    )
    args = parser.parse_args(argv)

    paths.set_output_root(args.output_dir)
    models_dir = paths.gb_models_dir()
    out = args.out or models_dir / "index.html"
    page = build_page(
        models_dir,
        args.catalogue.read_text(encoding="utf-8"),
        _registered_model_ids(),
        out.resolve().parent,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
