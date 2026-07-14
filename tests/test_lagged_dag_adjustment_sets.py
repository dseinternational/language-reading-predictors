# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""CI guard for the time-lagged DAG adjustment-set derivations (#250).

The lagged coupling models (LCSM-081/181/082) were designed against verified
d-separation derivations on ``dag/dag-language-reading-lagged.dagitty`` and a
crossover-aware three-slice unroll of the same structure
(``notes/202607141030-time-lagged-model-designs.md``). These tests re-run the
load-bearing checks so the claims cannot silently go stale when the DAG is
next revised: if an edit to the ``.dagitty`` breaks a test here, the
corresponding model's adjustment set (and the design note) needs re-deriving.

Every check removes latent general ability ``GA`` first — no measured set can
block it, so the assertions state the honest claim ("``GA`` aside, this set
blocks every backdoor") and the models label their couplings adjusted
associations accordingly.
"""

from __future__ import annotations

import re
from pathlib import Path

import networkx as nx
import pytest

REPO = Path(__file__).resolve().parents[1]
DAG_PATH = REPO / "dag" / "dag-language-reading-lagged.dagitty"

# ---------------------------------------------------------------------------
# The wave-slice structure (must mirror the .dagitty; the template test below
# cross-checks the mirror against the parsed file edge-for-edge).
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


def three_slice_unroll() -> nx.DiGraph:
    """Crossover-aware unroll: window 1 immediate-arm only, window 2 both arms."""
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
    for w in (1, 2):
        g.add_edge("IG", f"IS_{w}")
        g.add_edge("GA", f"IS_{w}")
        g.add_edge(f"A_{w}", f"IS_{w}")
        for tgt in ITT_TARGETS:
            g.add_edge("IG", f"{tgt}_{w + 1}")
            g.add_edge(f"IS_{w}", f"{tgt}_{w + 1}")
    return g


def blocks_backdoors(g: nx.DiGraph, x: str, y: str, z: set[str]) -> bool:
    """Backdoor validity of ``z`` for x -> y with latent GA removed (see module doc)."""
    h = g.copy()
    h.remove_edges_from(list(h.out_edges(x)))
    if "GA" in h:
        h.remove_node("GA")
    return nx.is_d_separator(h, {x}, {y}, set(z) - {"GA"})


@pytest.fixture(scope="module")
def template() -> nx.DiGraph:
    return parse_dagitty(DAG_PATH)


@pytest.fixture(scope="module")
def unrolled() -> nx.DiGraph:
    return three_slice_unroll()


def test_template_parses_and_is_acyclic(template):
    assert nx.is_directed_acyclic_graph(template)
    # The header records 36 nodes / 195 edges; a drift here means the .dagitty
    # was revised and every derivation below needs re-checking.
    assert template.number_of_nodes() == 36
    assert template.number_of_edges() == 195


def test_unroll_slices_mirror_the_dagitty_template(template):
    """The hand-coded slice structure must equal the parsed .dagitty slice.

    Guards the mirror itself: every within-wave, carry-over, reverse, age, HS
    and GA edge in the parsed two-slice template must appear in the unroll's
    corresponding slices (modulo the _t/_t1 vs _1/_2 naming and the
    intervention block, which the unroll extends to be crossover-aware).
    """
    unroll = three_slice_unroll()

    def rename(node: str) -> str:
        if node.endswith("_t1"):
            return node[:-3] + "_2"
        if node.endswith("_t"):
            return node[:-2] + "_1"
        return node

    for u, v in template.edges:
        if u in ("IG", "IS_t") or v == "IS_t":
            continue  # the unroll's intervention block is deliberately wider
        assert unroll.has_edge(rename(u), rename(v)), (u, v)


# --- the LCSM-081 estimands: W -> TE and W -> TR ---------------------------


def test_w_te_template_set_is_valid_on_transition_one(unrolled):
    assert blocks_backdoors(
        unrolled, "WR_1", "TE_2", {"TE_1", "TR_1", "RW_1", "SP_1", "A_1", "HS"}
    )


def test_arm_is_a_confounder_on_transition_two(unrolled):
    """The crossover result: the transition-1 set fails on transition 2 until
    IG is added — why LCSM-081/082 must carry arm x window intercepts."""
    base = {"TE_2", "TR_2", "RW_2", "SP_2", "A_2", "HS"}
    assert not blocks_backdoors(unrolled, "WR_2", "TE_3", base)
    assert blocks_backdoors(unrolled, "WR_2", "TE_3", base | {"IG"})


def test_lcsm_081_conditioning_ladder(unrolled):
    """The fitted model's conditioning set is exactly the verified set: each
    rung below the full block leaves a measured backdoor open."""
    assert not blocks_backdoors(unrolled, "WR_2", "TE_3", {"TE_2", "A_2", "IG"})
    assert not blocks_backdoors(
        unrolled, "WR_2", "TE_3", {"TE_2", "TR_2", "A_2", "IG"}
    )
    assert not blocks_backdoors(
        unrolled, "WR_2", "TE_3", {"TE_2", "TR_2", "A_2", "IG", "HS"}
    )
    full = {"TE_2", "TR_2", "A_2", "IG", "HS", "RW_2", "SP_2"}
    assert blocks_backdoors(unrolled, "WR_2", "TE_3", full)


def test_w_tr_sets_are_valid(unrolled):
    assert blocks_backdoors(
        unrolled, "WR_2", "TR_3", {"TR_2", "RW_2", "A_2", "HS", "IG"}
    )
    # The 081 model superset is also valid for the W -> TR coupling.
    assert blocks_backdoors(
        unrolled, "WR_2", "TR_3",
        {"TE_2", "TR_2", "RW_2", "SP_2", "A_2", "HS", "IG"},
    )


# --- why W -> PA is exploratory and W -> RW deferred ------------------------


def test_w_pa_and_w_rw_need_the_unfittable_wide_set(unrolled):
    small = {"PA_2", "LS_2", "A_2", "IG", "HS", "SP_2", "RW_2"}
    assert not blocks_backdoors(unrolled, "WR_2", "PA_3", small)
    wide = {
        "PA_2", "LS_2", "EV_2", "TE_2", "TR_2", "RV_2", "NW_2",
        "SP_2", "RW_2", "A_2", "HS", "IG", "WR_1",
    }
    assert blocks_backdoors(unrolled, "WR_2", "PA_3", wide)
    assert blocks_backdoors(unrolled, "WR_2", "RW_3", wide)


# --- the #264 mediation preview ---------------------------------------------


def test_mediation_te_worked_example(unrolled):
    """Baseline E/R are admissible but not required; the all-baseline style
    set does not strictly block the mediator-outcome backdoors."""
    baseline_style = {"IG", "A_2", "TE_1", "WR_1", "EV_1", "RV_1"}
    assert not blocks_backdoors(unrolled, "TE_2", "WR_2", baseline_style)
    contemporaneous = {
        "IG", "IS_1", "A_2", "TE_1", "WR_1", "HS",
        "TR_2", "RW_2", "SP_2", "LS_2",
    }
    assert blocks_backdoors(unrolled, "TE_2", "WR_2", contemporaneous)
    assert blocks_backdoors(
        unrolled, "TE_2", "WR_2", contemporaneous | {"EV_1", "RV_1"}
    )
