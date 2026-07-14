# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""d-separation checks behind ``notes/202607141030-time-lagged-model-designs.md``.

Two graphs are checked with :mod:`networkx`:

1. The two-slice template parsed directly from
   ``dag/dag-language-reading-lagged.dagitty`` (Option A, adopted 2026-07-13).
2. A crossover-aware **three-slice unroll** built from the same structure, in
   which the intervention is active in window 1 (immediate arm) *and* window 2
   (both arms) — the arm-specific timing the two-slice template cannot show.

Every ``[VALID]`` / ``[NOT-VALID]`` line quoted in the design note comes from
running this script. ``GA`` (latent general ability) is removed before each
check: no measured set can block it, so the checks ask the honest question —
"GA aside, does this set block every backdoor?" — and the note labels every
coupling an adjusted association accordingly.

Run with the project environment::

    python notes/assets/202607141030-lagged-dsep-checks.py
"""

from __future__ import annotations

import re
from pathlib import Path

import networkx as nx

REPO = Path(__file__).resolve().parents[2]
DAG_PATH = REPO / "dag" / "dag-language-reading-lagged.dagitty"


# ---------------------------------------------------------------------------
# Graph 1: the two-slice template, parsed from the .dagitty source of truth
# ---------------------------------------------------------------------------


def parse_dagitty(path: Path) -> nx.DiGraph:
    src = path.read_text()
    body = src[src.index("dag {") + len("dag {") : src.rindex("}")]
    g = nx.DiGraph()
    for line in body.splitlines():
        line = line.split("//")[0].strip()
        if not line or "->" not in line:
            continue
        m = re.match(r"^(\S+)\s*->\s*\{([^}]*)\}$", line)
        if m:
            for b in m.group(2).split():
                g.add_edge(m.group(1), b)
            continue
        m = re.match(r"^(\S+)\s*->\s*(\S+)$", line)
        if m:
            g.add_edge(m.group(1), m.group(2))
            continue
        raise ValueError(f"unparsed dagitty line: {line!r}")
    return g


# ---------------------------------------------------------------------------
# Graph 2: crossover-aware three-slice unroll of the same structure
# ---------------------------------------------------------------------------

SKILLS = [
    "TR", "TE", "RW", "RV", "EV", "LF", "RG", "EI", "EG",
    "SP", "LS", "NW", "PA", "PS", "WR",
]
WITHIN = [
    ("TR", ["TE", "RV", "EV", "LF", "RG", "WR"]),
    ("TE", ["EV", "EG", "EI", "PA", "WR"]),
    ("RW", ["TE", "EV", "TR", "RV", "PA", "NW", "PS"]),
    ("RV", ["EV", "LF", "RG", "WR"]),
    ("EV", ["EG", "EI", "PA", "WR"]),
    ("SP", ["TE", "EV", "LS", "PA", "NW"]),
    ("LS", ["NW", "PA", "PS", "WR"]),
    ("NW", ["WR"]),
    ("PA", ["NW", "WR", "PS"]),
    ("RG", ["EG"]),
]
REVERSE = ["TE", "TR", "PA", "RW"]  # WR_w -> {..}_{w+1}
HS_CHILDREN = ["TR", "RV", "TE", "EV", "SP", "RW", "PA", "LS"]
ITT_TARGETS = ["TR", "TE", "PA", "LS", "WR", "PS", "EI", "EG"]


def three_slice_unroll() -> nx.DiGraph:
    g = nx.DiGraph()
    waves = [1, 2, 3]
    for w in waves:
        for a, targets in WITHIN:
            for b in targets:
                g.add_edge(f"{a}_{w}", f"{b}_{w}")
        for s in SKILLS:
            g.add_edge("GA", f"{s}_{w}")
            g.add_edge(f"A_{w}", f"{s}_{w}")
        for s in HS_CHILDREN:
            g.add_edge("HS", f"{s}_{w}")
    for w in waves[:-1]:
        for s in SKILLS:
            g.add_edge(f"{s}_{w}", f"{s}_{w + 1}")
        g.add_edge(f"A_{w}", f"A_{w + 1}")
        for r in REVERSE:
            g.add_edge(f"WR_{w}", f"{r}_{w + 1}")
    # Crossover-aware intervention: window 1 (t1->t2) active for the immediate
    # arm; window 2 (t2->t3) active for BOTH arms (immediate continues block 2,
    # waitlist crosses over to block 1). IG is randomised at baseline and drives
    # the sessions in each window.
    for w in (1, 2):
        g.add_edge("IG", f"IS_{w}")
        g.add_edge("GA", f"IS_{w}")
        g.add_edge(f"A_{w}", f"IS_{w}")
        for tgt in ITT_TARGETS:
            g.add_edge("IG", f"{tgt}_{w + 1}")
            g.add_edge(f"IS_{w}", f"{tgt}_{w + 1}")
    return g


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def backdoor_graph(g: nx.DiGraph, x: str) -> nx.DiGraph:
    h = g.copy()
    h.remove_edges_from(list(h.out_edges(x)))
    return h


def check(g: nx.DiGraph, x: str, y: str, z: set[str], label: str = "") -> bool:
    """Backdoor validity of z for x -> y, with latent GA removed (see module doc)."""
    h = backdoor_graph(g, x)
    if "GA" in h:
        h.remove_node("GA")
    z = set(z) - {"GA"}
    ok = nx.is_d_separator(h, {x}, {y}, z)
    print(f"  [{'VALID' if ok else 'NOT-VALID'}] {x} -> {y} | {sorted(z)} {label}")
    return ok


def main() -> None:
    tmpl = parse_dagitty(DAG_PATH)
    assert nx.is_directed_acyclic_graph(tmpl)
    print(f"template: {tmpl.number_of_nodes()} nodes, {tmpl.number_of_edges()} edges")

    print("\n-- two-slice template: W -> TE / TR (transition 1 semantics) --")
    check(tmpl, "WR_t", "TE_t1", {"TE_t", "TR_t", "RW_t", "SP_t", "A_t", "HS"})
    check(tmpl, "WR_t", "TR_t1", {"TR_t", "RW_t", "A_t", "HS"})

    g3 = three_slice_unroll()
    assert nx.is_directed_acyclic_graph(g3)
    print(f"\n3-slice unroll: {g3.number_of_nodes()} nodes, {g3.number_of_edges()} edges")

    print("\n-- IG becomes a confounder on transition 2 (the crossover) --")
    base2 = {"TE_2", "TR_2", "RW_2", "SP_2", "A_2", "HS"}
    check(g3, "WR_2", "TE_3", base2, "(no IG)")
    check(g3, "WR_2", "TE_3", base2 | {"IG"}, "(+ IG)")
    check(g3, "WR_1", "TE_2", {"TE_1", "TR_1", "RW_1", "SP_1", "A_1", "HS"}, "(transition 1: no IG needed)")

    print("\n-- the practical sets quoted in the design note (transition 2) --")
    check(g3, "WR_2", "TE_3", {"TE_2", "TR_2", "RW_2", "SP_2", "A_2", "HS", "IG"}, "(W->TE)")
    check(g3, "WR_2", "TR_3", {"TR_2", "RW_2", "A_2", "HS", "IG"}, "(W->TR)")
    check(g3, "WR_2", "TR_3", {"TE_2", "TR_2", "RW_2", "SP_2", "A_2", "HS", "IG"}, "(W->TR, model superset)")

    print("\n-- what the fitted LCSM-081 conditions on, stepping up to the full set --")
    check(g3, "WR_2", "TE_3", {"TE_2", "A_2", "IG"}, "(TE process + age + arm-window)")
    check(g3, "WR_2", "TE_3", {"TE_2", "TR_2", "A_2", "IG"}, "(+ TR process)")
    check(g3, "WR_2", "TE_3", {"TE_2", "TR_2", "A_2", "IG", "HS"}, "(+ hs covariate)")
    check(g3, "WR_2", "TE_3", {"TE_2", "TR_2", "A_2", "IG", "HS", "RW_2", "SP_2"}, "(+ rw/sp covariates: full)")

    print("\n-- W -> PA and W -> RW need a far wider set (hence exploratory / deferred) --")
    wide = {"PA_2", "LS_2", "EV_2", "TE_2", "TR_2", "RV_2", "NW_2", "SP_2", "RW_2", "A_2", "HS", "IG", "WR_1"}
    check(g3, "WR_2", "PA_3", {"PA_2", "LS_2", "A_2", "IG", "HS", "SP_2", "RW_2"}, "(an 081-sized set fails)")
    check(g3, "WR_2", "PA_3", wide, "(all-measured incl. floored NW_2 and WR_1)")
    check(g3, "WR_2", "RW_3", wide, "(same for W->RW)")

    print("\n-- #264 preview: mediation M=TE_2 -> Y=WR_2 in the t1->t2 window --")
    check(g3, "TE_2", "WR_2", {"IG", "A_2", "TE_1", "WR_1", "EV_1", "RV_1"}, "(baseline-style set)")
    check(
        g3, "TE_2", "WR_2",
        {"IG", "IS_1", "A_2", "TE_1", "WR_1", "HS", "TR_2", "RW_2", "SP_2", "LS_2", "EV_1", "RV_1"},
        "(+ contemporaneous non-descendant states)",
    )
    check(
        g3, "TE_2", "WR_2",
        {"IG", "IS_1", "A_2", "TE_1", "WR_1", "HS", "TR_2", "RW_2", "SP_2", "LS_2"},
        "(same WITHOUT baseline vocab E/R)",
    )


if __name__ == "__main__":
    main()
