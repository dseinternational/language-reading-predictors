# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""CI guard for the time-lagged DAG adjustment-set derivations (#250, #264).

The lagged coupling models (LCSM-081/181/082) were designed against verified
d-separation derivations on ``dag/dag-language-reading-lagged.dagitty`` and a
crossover-aware three-slice unroll of the same structure
(``notes/202607141030-time-lagged-model-designs.md``), and the mediation
family's adjustment sets were settled against the same unroll
(``notes/202607142340-lrp264-mediation-adjustment-dsep.md``). These tests
re-run the load-bearing checks so the claims cannot silently go stale when the
DAG is next revised: if an edit to the ``.dagitty`` (or to a MED model's
``SPEC.adjustment``) breaks a test here, the corresponding adjustment set (and
the derivation note) needs re-deriving.

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


def unroll(n_waves: int) -> nx.DiGraph:
    """Crossover-aware unroll over ``n_waves`` waves.

    Sessions run in every inter-wave window the unroll covers (immediate arm
    from window 1, waitlist from window 2, both arms continuing in window 3 —
    the delivery recorded in the #250 design note), so each window ``w`` gets
    ``IG -> IS_w`` and ``IS_w`` / ``IG`` into the ITT targets at wave w+1.
    ``unroll(3)`` is the design note's three-slice graph (window 1
    immediate-arm only, window 2 both arms).
    """
    g = nx.DiGraph()
    waves = list(range(1, n_waves + 1))
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
        g.add_edge("IG", f"IS_{w}")
        g.add_edge("GA", f"IS_{w}")
        g.add_edge(f"A_{w}", f"IS_{w}")
        for tgt in ITT_TARGETS:
            g.add_edge("IG", f"{tgt}_{w + 1}")
            g.add_edge(f"IS_{w}", f"{tgt}_{w + 1}")
    return g


def three_slice_unroll() -> nx.DiGraph:
    return unroll(3)


def blocks_backdoors(g: nx.DiGraph, x: str | set[str], y: str, z: set[str]) -> bool:
    """Backdoor validity of ``z`` for x -> y with latent GA removed (see module doc).

    ``x`` may be a set (a joint mediator block): the out-edges of every member
    are removed and the whole block must be d-separated from ``y``.
    """
    xs = {x} if isinstance(x, str) else set(x)
    h = g.copy()
    for xi in xs:
        h.remove_edges_from(list(h.out_edges(xi)))
    if "GA" in h:
        h.remove_node("GA")
    return nx.is_d_separator(h, xs, {y}, set(z) - {"GA"})


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


# --- the #264 mediation-family settlement ------------------------------------
#
# Per-model derivations from notes/202607142340-lrp264-mediation-adjustment-dsep.md
# (script: notes/assets/202607142340-med-adjustment-dsep.py). The adjustment
# sets are read from the live ``SPEC`` objects so an edit to any MED model's
# set re-triggers the derivation here.

# ModelSpec symbol -> DAG node (the node key in the .dagitty header).
SYMBOL_TO_NODE = {
    "W": "WR", "L": "LS", "B": "PA", "N": "NW", "E": "EV", "R": "RV",
    "T": "RG", "TE": "TE", "TR": "TR",
}
# Raw adjuster covariates -> DAG node (entered at baseline in the fitted models).
COVARIATE_TO_NODE = {"hs": "HS", "deapp_c": "SP", "erbto": "RW"}

# A collider-free mediator->outcome backdoor path per model, every one of whose
# interior nodes is an IG-descendant — the proof that NO treatment-non-descendant
# set can block the mediator-outcome confounding (natural effects stay
# unidentified whatever happens to E/R).
MED_WITNESSES = {
    "lrp_rli_med_059": ["LS_2", "<-", "IS_1", "->", "WR_2"],
    "lrp_rli_med_062": ["LS_2", "<-", "IS_1", "->", "WR_2"],
    "lrp_rli_med_064": ["LS_2", "<-", "IS_1", "->", "WR_2"],
    "lrp_rli_med_066": ["LS_2", "<-", "IS_1", "->", "WR_2"],
    "lrp_rli_med_068": ["TE_2", "<-", "IS_1", "->", "WR_2"],
    "lrp_rli_med_074": ["NW_2", "<-", "LS_2", "<-", "IS_1", "->", "WR_2"],
    "lrp_rli_med_075": ["LS_2", "<-", "IS_1", "->", "WR_2"],
    "lrp_rli_med_076": ["LS_2", "<-", "IS_1", "->", "WR_2", "->", "WR_3", "->", "WR_4"],
    "lrp_rli_med_078": ["LS_2", "<-", "IS_1", "->", "WR_2"],
    "lrp_rli_med_079": ["RG_2", "<-", "TR_2", "->", "WR_2"],
    "lrp_rli_med_080": ["TR_2", "<-", "IS_1", "->", "WR_2"],
    "lrp_rli_med_086": ["LS_2", "<-", "IS_1", "->", "PA_2", "->", "NW_2"],
    "lrp_rli_med_087": ["LS_2", "<-", "IS_1", "->", "PA_2"],
    "lrp_rli_med_186": ["LS_2", "<-", "IS_1", "->", "PA_2", "->", "NW_2"],
    "lrp_rli_med_187": ["LS_2", "<-", "IS_1", "->", "PA_2"],
}

# Expected role of baseline E / R per model ("member": in the valid parent set
# pa(M); "proxy": the contemporaneous EV_2/RV_2 is in pa(M) and the baseline is
# its admissible pre-treatment proxy; "precision": on no minimal backdoor route).
MED_ER_ROLES = {
    "lrp_rli_med_059": ("precision", "precision"),
    "lrp_rli_med_062": ("proxy", "precision"),
    "lrp_rli_med_064": ("member", "proxy"),
    "lrp_rli_med_066": ("proxy", "precision"),
    "lrp_rli_med_068": ("precision", "precision"),
    "lrp_rli_med_074": ("precision", "precision"),
    "lrp_rli_med_075": ("proxy", "precision"),
    "lrp_rli_med_076": ("precision", "precision"),
    "lrp_rli_med_078": ("precision", "precision"),
    "lrp_rli_med_079": ("precision", "proxy"),
    "lrp_rli_med_080": ("precision", "precision"),
    "lrp_rli_med_086": ("precision", "precision"),
    "lrp_rli_med_087": ("precision", "precision"),
    "lrp_rli_med_186": ("precision", "precision"),
    "lrp_rli_med_187": ("precision", "precision"),
}


@pytest.fixture(scope="module")
def unrolled4() -> nx.DiGraph:
    return unroll(4)


@pytest.fixture(scope="module")
def med_specs():
    import importlib

    return {
        name: importlib.import_module(
            f"language_reading_predictors.statistical_models.{name}"
        ).SPEC
        for name in MED_WITNESSES
    }


def _map_adjustment(entries, outcome_symbol: str) -> set[str]:
    """Map a ``ModelSpec.adjustment`` list to unrolled-graph nodes.

    Baselines land at wave 1; ``W_pre`` is the outcome-own-baseline marker
    (MED-087 uses it for B's pre-score); ``A`` is granted {A_1, A_2} because
    age at later waves is deterministic given wave spacing; ``*_missing``
    indicators are not graph nodes.
    """
    nodes: set[str] = set()
    for e in entries:
        if e.endswith("_missing"):
            continue
        if e == "G":
            nodes.add("IG")
        elif e == "A":
            nodes.update({"A_1", "A_2"})
        elif e == "W_pre":
            nodes.add(f"{SYMBOL_TO_NODE[outcome_symbol]}_1")
        elif e.endswith("_t1"):
            nodes.add(f"{SYMBOL_TO_NODE[e[:-3]]}_1")
        elif e in COVARIATE_TO_NODE:
            n = COVARIATE_TO_NODE[e]
            nodes.add("HS" if n == "HS" else f"{n}_1")
        elif e in SYMBOL_TO_NODE:
            nodes.add(f"{SYMBOL_TO_NODE[e]}_1")
        else:
            raise ValueError(f"unmapped adjustment entry: {e!r}")
    return nodes


def _med_case(spec, unrolled, unrolled4):
    """(graph, mediator nodes, outcome node, mapped fitted set) for a MED spec."""
    if spec.mechanism_symbol is not None:
        mediator_symbols = (spec.mechanism_symbol,)
    else:
        mediator_symbols = spec.extra.get("mediators") or spec.extra["route_symbols"]
    outcome_wave = spec.extra.get("outcome_time", 2)
    g = unrolled4 if outcome_wave == 4 else unrolled
    ms = {f"{SYMBOL_TO_NODE[s]}_2" for s in mediator_symbols}
    y = f"{SYMBOL_TO_NODE[spec.outcome_symbol]}_{outcome_wave}"
    c = _map_adjustment(spec.adjustment, spec.outcome_symbol)
    return g, ms, y, c


def _parent_set(g: nx.DiGraph, mediators: set[str]) -> set[str]:
    ps: set[str] = set()
    for m in mediators:
        ps |= set(g.predecessors(m))
    return (ps - {"GA"}) - mediators


def test_baseline_vocabulary_precedes_treatment(unrolled, unrolled4):
    """The #259 descendant argument fails with measurement occasions explicit:
    baseline E/R (EV_1/RV_1) are not descendants of the randomised IG."""
    for g in (unrolled, unrolled4):
        ig_desc = nx.descendants(g, "IG")
        assert "EV_1" not in ig_desc
        assert "RV_1" not in ig_desc


@pytest.mark.parametrize("name", sorted(MED_WITNESSES))
def test_med_fitted_set_is_admissible_and_e_r_settle_nothing(
    name, med_specs, unrolled, unrolled4
):
    """The fitted all-baseline set contains no IG-descendant (cross-world
    admissible), does not strictly block the mediator-outcome backdoors, and
    its status is unchanged by dropping E/R — retention is a precision call,
    not an identification one."""
    g, ms, y, c = _med_case(med_specs[name], unrolled, unrolled4)
    ig_desc = nx.descendants(g, "IG")
    assert not (c & ig_desc), f"{name}: fitted set contains IG-descendants"
    assert not blocks_backdoors(g, ms, y, c)
    assert not blocks_backdoors(g, ms, y, c - {"EV_1", "RV_1"})


@pytest.mark.parametrize("name", sorted(MED_WITNESSES))
def test_med_witness_backdoor_defeats_every_admissible_set(
    name, med_specs, unrolled, unrolled4
):
    """Each model has a collider-free mediator-outcome backdoor whose interior
    nodes are all IG-descendants, so no treatment-non-descendant adjustment
    set can block it: natural effects stay unidentified regardless of E/R."""
    g, ms, y, _ = _med_case(med_specs[name], unrolled, unrolled4)
    path = MED_WITNESSES[name]
    nodes, arrows = path[0::2], path[1::2]
    assert nodes[0] in ms and nodes[-1] == y
    ig_desc = nx.descendants(g, "IG")
    for i, arr in enumerate(arrows):
        u, v = nodes[i], nodes[i + 1]
        assert g.has_edge(u, v) if arr == "->" else g.has_edge(v, u), (u, arr, v)
    for i in range(1, len(nodes) - 1):
        assert not (arrows[i - 1] == "->" and arrows[i] == "<-"), (
            f"witness has a collider at {nodes[i]}"
        )
        assert nodes[i] in ig_desc, f"{nodes[i]} is not an IG-descendant"


@pytest.mark.parametrize("name", sorted(MED_WITNESSES))
def test_med_parent_set_is_valid_and_e_r_are_harmless(
    name, med_specs, unrolled, unrolled4
):
    """pa(M) (descendants allowed, GA aside) strictly blocks the backdoors and
    stays valid with baseline E/R added — E/R are admissible and harmless; the
    E/R roles the note records are re-derived from the graph."""
    g, ms, y, _ = _med_case(med_specs[name], unrolled, unrolled4)
    pa = _parent_set(g, ms)
    assert blocks_backdoors(g, ms, y, pa)
    assert blocks_backdoors(g, ms, y, pa | {"EV_1", "RV_1"})
    for base, contemp, expected in (
        ("EV_1", "EV_2", MED_ER_ROLES[name][0]),
        ("RV_1", "RV_2", MED_ER_ROLES[name][1]),
    ):
        if base in pa:
            role = "member"
        elif contemp in pa:
            role = "proxy"
        else:
            role = "precision"
        assert role == expected, f"{name}: {base} role {role} != {expected}"
