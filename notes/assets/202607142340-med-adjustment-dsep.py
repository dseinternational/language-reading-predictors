# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Time-indexed d-separation for the mediation-family adjustment sets (#264).

Extends ``notes/assets/202607141030-lagged-dsep-checks.py`` (the #250 script)
from the LCSM couplings to the full mediation family: for each of the thirteen
MED models the crossover-aware unroll of ``dag/dag-language-reading-lagged.dagitty``
is used to derive, mechanically:

1. whether the fitted (all-baseline) adjustment set contains any descendant of
   the randomised treatment ``IG`` (cross-world admissibility of the set itself);
2. whether baseline vocabulary ``EV_1`` / ``RV_1`` (the models' ``E`` / ``R``)
   are descendants of ``IG`` or of any mediator (the #259 claim under test);
3. whether the fitted set blocks the mediator -> outcome backdoors (``GA`` aside);
4. a **witness path** per model — a collider-free backdoor path every one of
   whose interior nodes is a descendant of ``IG`` — proving no treatment-
   non-descendant set can block the mediator-outcome confounding at all;
5. the strictly-valid parent set ``pa(M)`` (descendants allowed), that it stays
   valid with ``EV_1``/``RV_1`` added (harmlessness), and which of its members
   are ``IG``-descendants (the recanting witnesses, named);
6. the per-model role of ``E``/``R``: member of the valid parent set, admissible
   proxy for a treatment-affected contemporaneous confounder, or precision-only.

``GA`` (latent general ability) is removed before every d-separation check: no
measured set can block it, so the checks ask the honest question ("GA aside,
does this set block every backdoor?") and the family labels its decompositions
adjusted associations / g-formula-under-assumptions accordingly.

Age enters the fitted models once, at baseline; on the unroll the fitted ``A``
is granted {A_1, A_2} because age at later waves is deterministic given wave
spacing (the .dagitty header records the A_t -> A_t1 edge as "a placeholder for
maturation, not a stochastic cause").

Run with the project environment::

    python notes/assets/202607142340-med-adjustment-dsep.py
"""

from __future__ import annotations

import re
from pathlib import Path

import networkx as nx

REPO = Path(__file__).resolve().parents[2]
DAG_PATH = REPO / "dag" / "dag-language-reading-lagged.dagitty"


# ---------------------------------------------------------------------------
# The wave-slice structure (mirrors the .dagitty; identical to the #250 script)
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
    the delivery recorded in the #250 design note), so each window w gets
    IG -> IS_w and IS_w / IG -> the ITT targets at wave w+1.
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


# ---------------------------------------------------------------------------
# The mediation family, mapped onto the unroll
# ---------------------------------------------------------------------------

# ModelSpec symbol -> DAG node (same mapping as measures.py / the DAG node key).
SYMBOL_TO_NODE = {
    "W": "WR", "L": "LS", "B": "PA", "N": "NW", "E": "EV", "R": "RV",
    "T": "RG", "TE": "TE", "TR": "TR",
}
# Raw adjuster covariates -> DAG node (taken at baseline in the fitted models).
COVARIATE_TO_NODE = {"hs": "HS", "deapp_c": "SP", "erbto": "RW"}


def map_adjustment(entries: list[str], outcome_symbol: str) -> set[str]:
    """Map a ModelSpec.adjustment list to unrolled-graph nodes.

    Baselines land at wave 1; ``W_pre`` is the outcome-own-baseline marker
    (MED-087 uses it for B's pre-score); ``A`` is granted {A_1, A_2} by age
    determinism; ``*_missing`` indicators are not graph nodes.
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


# Witness paths: alternating node / arrow sequences from a mediator to the
# outcome. Every interior node must be an IG-descendant (asserted below).
MODELS: list[dict] = [
    dict(mid="med-059", mediators=["L"], outcome="W",
         adjustment=["G", "A", "E", "R", "L_t1", "W_pre", "hs", "deapp_c"],
         witness=["LS_2", "<-", "IS_1", "->", "WR_2"]),
    dict(mid="med-062", mediators=["L", "B"], outcome="W",
         adjustment=["G", "A", "E", "R", "W_pre", "hs", "erbto", "deapp_c"],
         witness=["LS_2", "<-", "IS_1", "->", "WR_2"]),
    dict(mid="med-064", mediators=["L", "E"], outcome="W",
         adjustment=["G", "A", "R", "W_pre", "L_t1", "E_t1", "hs", "erbto", "deapp_c"],
         witness=["LS_2", "<-", "IS_1", "->", "WR_2"]),
    dict(mid="med-066", mediators=["L", "B"], outcome="W",
         adjustment=["G", "A", "E", "R", "W_pre", "L_t1", "B_t1"],
         witness=["LS_2", "<-", "IS_1", "->", "WR_2"]),
    dict(mid="med-068", mediators=["TE"], outcome="W",
         adjustment=["G", "A", "L", "R", "W_pre", "TE_t1"],
         witness=["TE_2", "<-", "IS_1", "->", "WR_2"]),
    dict(mid="med-074", mediators=["N"], outcome="W",
         adjustment=["G", "A", "E", "R", "W_pre", "N_t1"],
         witness=["NW_2", "<-", "LS_2", "<-", "IS_1", "->", "WR_2"]),
    dict(mid="med-075", mediators=["L", "B"], outcome="W",
         adjustment=["G", "A", "E", "R", "W_pre", "L_t1", "B_t1"],
         witness=["LS_2", "<-", "IS_1", "->", "WR_2"]),
    dict(mid="med-076", mediators=["L"], outcome="W", outcome_wave=4,
         adjustment=["G", "A", "E", "R", "W_pre", "L_t1"],
         witness=["LS_2", "<-", "IS_1", "->", "WR_2", "->", "WR_3", "->", "WR_4"]),
    dict(mid="med-078", mediators=["L"], outcome="W",
         adjustment=["G", "A", "E", "R", "L_t1", "W_pre", "hs", "deapp_c"],
         witness=["LS_2", "<-", "IS_1", "->", "WR_2"]),
    dict(mid="med-079", mediators=["T"], outcome="W",
         adjustment=["G", "A", "E", "R", "W_pre", "T_t1"],
         witness=["RG_2", "<-", "TR_2", "->", "WR_2"]),
    dict(mid="med-080", mediators=["TR"], outcome="W",
         adjustment=["G", "A", "L", "E", "W_pre", "TR_t1"],
         witness=["TR_2", "<-", "IS_1", "->", "WR_2"]),
    dict(mid="med-086", mediators=["L"], outcome="N",
         adjustment=["G", "A", "L_t1", "B", "hs", "deapp_c", "erbto"],
         witness=["LS_2", "<-", "IS_1", "->", "PA_2", "->", "NW_2"]),
    dict(mid="med-087", mediators=["L"], outcome="B",
         adjustment=["G", "A", "W_pre", "L_t1", "hs", "deapp_c"],
         witness=["LS_2", "<-", "IS_1", "->", "PA_2"]),
]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def blocks_backdoors(g: nx.DiGraph, xs: set[str], y: str, z: set[str]) -> bool:
    """Backdoor validity of z for the (joint) mediator set xs -> y, GA removed."""
    h = g.copy()
    for x in xs:
        h.remove_edges_from(list(h.out_edges(x)))
    if "GA" in h:
        h.remove_node("GA")
    return nx.is_d_separator(h, set(xs), {y}, set(z) - {"GA"})


def verify_witness(g: nx.DiGraph, path: list[str], ig_desc: set[str]) -> list[str]:
    """Assert the witness path exists, is collider-free, and every interior
    node is an IG-descendant; return the interior nodes."""
    nodes = path[0::2]
    arrows = path[1::2]
    assert len(arrows) == len(nodes) - 1
    for i, arr in enumerate(arrows):
        u, v = nodes[i], nodes[i + 1]
        if arr == "->":
            assert g.has_edge(u, v), f"missing edge {u} -> {v}"
        else:
            assert g.has_edge(v, u), f"missing edge {v} -> {u}"
    # collider at interior position i iff the arrows either side both point in
    for i in range(1, len(nodes) - 1):
        assert not (arrows[i - 1] == "->" and arrows[i] == "<-"), (
            f"witness has a collider at {nodes[i]}"
        )
    interior = nodes[1:-1]
    for n in interior:
        assert n in ig_desc, f"witness interior {n} is not an IG-descendant"
    return interior


def parent_set(g: nx.DiGraph, mediators: set[str]) -> set[str]:
    ps: set[str] = set()
    for m in mediators:
        ps |= set(g.predecessors(m))
    return (ps - {"GA"}) - mediators


def main() -> None:
    tmpl = parse_dagitty(DAG_PATH)
    assert nx.is_directed_acyclic_graph(tmpl)
    assert tmpl.number_of_nodes() == 36 and tmpl.number_of_edges() == 195
    print(f"template ok: {tmpl.number_of_nodes()} nodes, {tmpl.number_of_edges()} edges")

    g3 = unroll(3)
    g4 = unroll(4)
    assert nx.is_directed_acyclic_graph(g3) and nx.is_directed_acyclic_graph(g4)

    for g, name in ((g3, "3-slice"), (g4, "4-slice")):
        ig_desc = nx.descendants(g, "IG")
        print(f"{name}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges; "
              f"|desc(IG)| = {len(ig_desc)}")
        # The #259 claim under test: baseline vocabulary precedes treatment.
        assert "EV_1" not in ig_desc and "RV_1" not in ig_desc
    print("EV_1 / RV_1 are NOT descendants of IG on any unroll "
          "(the #259 descendant argument fails)\n")

    for spec in MODELS:
        mid = spec["mid"]
        g = g4 if spec.get("outcome_wave") == 4 else g3
        ig_desc = nx.descendants(g, "IG")
        y = f"{SYMBOL_TO_NODE[spec['outcome']]}_{spec.get('outcome_wave', 2)}"
        ms = {f"{SYMBOL_TO_NODE[s]}_2" for s in spec["mediators"]}
        c = map_adjustment(spec["adjustment"], spec["outcome"])
        er_in_c = sorted(c & {"EV_1", "RV_1"})

        print(f"== {mid}: M = {sorted(ms)} -> Y = {y}")
        print(f"   fitted set (mapped): {sorted(c)}")

        # 1. cross-world admissibility of the fitted set itself
        bad = sorted(c & ig_desc)
        assert not bad, f"{mid}: fitted set contains IG-descendants {bad}"
        print("   [OK] fitted set contains no IG-descendant (cross-world-admissible)")

        # 2. E/R are not descendants of IG or of any mediator
        for n in ("EV_1", "RV_1"):
            assert n not in ig_desc
            for m in ms:
                assert n not in nx.descendants(g, m)

        # 3. does the fitted (all-baseline) set block the M -> Y backdoors?
        ok_c = blocks_backdoors(g, ms, y, c)
        ok_c_no_er = blocks_backdoors(g, ms, y, c - {"EV_1", "RV_1"})
        print(f"   [{'VALID' if ok_c else 'NOT-VALID'}] M -> Y | fitted set")
        if er_in_c:
            print(f"   [{'VALID' if ok_c_no_er else 'NOT-VALID'}] "
                  f"M -> Y | fitted set minus {er_in_c} (dropping E/R changes nothing)")
            assert ok_c == ok_c_no_er

        # 4. witness: a collider-free backdoor path blockable only at IG-descendants
        interior = verify_witness(g, spec["witness"], ig_desc)
        print(f"   witness path {' '.join(spec['witness'])}: interior {interior} "
              "are all IG-descendants -> NO treatment-non-descendant set can block it")

        # 5. the strictly-valid parent set (descendants allowed), and E/R harmlessness
        pa = parent_set(g, ms)
        assert blocks_backdoors(g, ms, y, pa), f"{mid}: pa(M) not valid?!"
        assert blocks_backdoors(g, ms, y, pa | {"EV_1", "RV_1"}), (
            f"{mid}: adding EV_1/RV_1 broke pa(M)!"
        )
        witnesses = sorted(pa & ig_desc)
        print(f"   pa(M) = {sorted(pa)}")
        print("   [VALID] M -> Y | pa(M), and [VALID] with EV_1/RV_1 added "
              "(E/R are harmless)")
        print(f"   recanting witnesses (IG-descendants in pa(M)): {witnesses}")

        # 6. the per-model role of E/R
        for base, contemp, sym in (("EV_1", "EV_2", "E"), ("RV_1", "RV_2", "R")):
            if base in pa:
                role = "member of the valid parent set (parent of a mediator)"
            elif contemp in pa:
                role = (f"admissible baseline proxy for {contemp}, a treatment-"
                        "affected member of the valid set")
            else:
                role = "precision-only (on no minimal backdoor route)"
            flag = "in fitted set" if base in c else "NOT in fitted set"
            print(f"   {sym} ({base}, {flag}): {role}")
        print()


if __name__ == "__main__":
    main()
