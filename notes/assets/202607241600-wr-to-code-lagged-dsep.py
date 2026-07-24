# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Can a lagged model test WORD READING -> later gains in LS / PA / NW?

Supports ``notes/202607241600-findings-word-reading-bands.md`` (part 3). The
descriptive pass in ``202607241600-word-reading-band-comparisons.py`` shows
word-reading standing tracking later movement in nonword reading and blending but
not letter sounds; this script asks the prior question — **which of those three
couplings is identifiable at all**, and with what adjustment set.

Same machinery and conventions as ``202607141030-lagged-dsep-checks.py`` (the
d-separation checks behind the LCSM-081/082 design note), with two corrections:

1. That script's ``REVERSE`` list is ``["TE", "TR", "PA", "RW"]`` — it predates the
   **2026-07-17** addition of ``WR_t -> LS_t1`` to
   ``dag/dag-language-reading-lagged.dagitty`` (added for the LRP-RLI-MED-176
   direction contrast). Re-run today it therefore builds a graph the DAG no longer
   matches. Note this is a stale *archived asset*, NOT an unguarded invariant: the
   design note's recommendation to promote the checks to a pytest was actioned, and
   ``tests/test_lagged_dag_adjustment_sets.py`` carries the corrected five-edge list
   plus a mirror assertion against the parsed ``.dagitty``. This script rebuilds the
   unroll from an explicit reverse-edge list and asserts that list against the
   ``.dagitty`` too, so the same drift cannot recur here.
2. It adds minimal-sufficient-set SEARCH rather than checking hand-written sets, so
   "no fittable set exists" is a derived result rather than an assertion.

``GA`` (latent general ability) is removed before every check: no measured set can
block it, so each check asks the honest question — "GA aside, does this set block
every backdoor?" — and every coupling is an **adjusted association**, never causal.

Run with the project environment::

    python notes/assets/202607241600-wr-to-code-lagged-dsep.py
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import networkx as nx

REPO = Path(__file__).resolve().parents[2]
DAG_PATH = REPO / "dag" / "dag-language-reading-lagged.dagitty"

# Rough per-parameter budget. n ~ 54 children x 3 transitions ~ 160 change rows, but
# the effective n for a between-child coupling is the 54 children. A latent-process
# model already spends ~12 parameters per process, so an adjustment set needing more
# than about six measured nodes beyond the target's own process is not fittable here.
FITTABLE_MAX_ADJUSTERS = 6

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
# The reverse (lagged) edges out of word reading, as at 2026-07-17. Asserted against
# the .dagitty source below — do not edit one without the other.
REVERSE = ["TE", "TR", "PA", "RW", "LS"]
HS_CHILDREN = ["TR", "RV", "TE", "EV", "SP", "RW", "PA", "LS"]
ITT_TARGETS = ["TR", "TE", "PA", "LS", "WR", "PS", "EI", "EG"]

# Measures with a registered Beta-Binomial denominator and usable data. RW (erb*) has
# no registered Measure and an out-of-range value (design note, 2026-07-14), so a
# model cannot condition on it as a *process* — it enters only as the standardised
# `erbto` covariate. Kept separate so "fittable" means fittable with real columns.
UNMEASURED = {"RW"}  # available as a covariate, not as a modelled process
LATENT = {"GA"}


def parse_dagitty(path: Path) -> nx.DiGraph:
    src = path.read_text()
    body = src[src.index("dag {") + len("dag {"): src.rindex("}")]
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


def assert_reverse_edges_match_dag(template: nx.DiGraph) -> None:
    """Fail loudly if REVERSE has drifted from the .dagitty source of truth."""
    # WR_t -> WR_t1 is the carry-over edge every measure has, not a reverse edge;
    # the unroll adds it in its own loop, so exclude it from the comparison.
    actual = {
        t[: -len("_t1")]
        for _, t in template.out_edges("WR_t")
        if t.endswith("_t1") and t != "WR_t1"
    }
    expected = set(REVERSE)
    if actual != expected:
        raise AssertionError(
            "REVERSE has drifted from dag-language-reading-lagged.dagitty.\n"
            f"  in the DAG but not in REVERSE: {sorted(actual - expected)}\n"
            f"  in REVERSE but not in the DAG: {sorted(expected - actual)}\n"
            "Update REVERSE (and re-derive every result in the supporting note)."
        )
    print(f"reverse edges WR_t -> *_t1 match the DAG: {sorted(actual)}")


def three_slice_unroll() -> nx.DiGraph:
    """Crossover-aware three-slice unroll (window 1 = immediate arm only; window 2 = both)."""
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


def backdoor_graph(g: nx.DiGraph, x: str) -> nx.DiGraph:
    h = g.copy()
    h.remove_edges_from(list(h.out_edges(x)))
    return h


def is_valid(g: nx.DiGraph, x: str, y: str, z: set[str]) -> bool:
    """Backdoor validity of z for x -> y, latent GA removed (see module docstring)."""
    h = backdoor_graph(g, x)
    h.remove_nodes_from(LATENT & set(h.nodes))
    return nx.is_d_separator(h, {x}, {y}, set(z) - LATENT)


def check(g: nx.DiGraph, x: str, y: str, z: set[str], label: str = "") -> bool:
    ok = is_valid(g, x, y, z)
    print(f"  [{'VALID' if ok else 'NOT-VALID'}] {x} -> {y} | {sorted(z)} {label}")
    return ok


def candidate_adjusters(g: nx.DiGraph, x: str, y: str) -> list[str]:
    """Measured non-descendants of x, excluding x and y.

    Descendants of x are excluded because conditioning on them either blocks the
    very path being estimated or opens a collider; ``IS`` (sessions) is excluded on
    the locked DAG's ID-3 result — it is a collider of arm and latent ability, so
    conditioning on it reopens the ability backdoor (design note, 2026-07-14).
    """
    desc = nx.descendants(g, x) | {x}
    return sorted(
        n for n in g.nodes
        if n not in desc and n != y
        and n not in LATENT
        and not n.startswith("IS_")
    )


def minimal_sets(g: nx.DiGraph, x: str, y: str, n_orders: int = 40,
                 seed: int = 20260724) -> list[frozenset[str]]:
    """Minimal sufficient adjustment sets, found by greedy pruning from pa(y).

    Exhaustive enumeration is infeasible here — the three-slice unroll leaves ~30
    candidate adjusters, so searching every subset up to size 8 is ~6e6 d-separation
    checks on a 50-node graph. Instead: start from the parent set of ``y`` (always
    sufficient in a DAG, latent ``GA`` aside), then repeatedly try to drop one
    element at a time, keeping the drop whenever the set stays valid. A set that
    survives is minimal by construction — no single element can be removed. Repeating
    over randomised drop orders surfaces genuinely different minimal sets, since
    which one you land in depends on the order you prune.

    This finds minimal sets, not necessarily the globally *smallest*; the reported
    sizes are therefore upper bounds on the minimum. That is the safe direction for
    the question being asked — if even the pruned set is too wide to fit, the
    coupling is unfittable regardless.
    """
    rng = random.Random(seed)
    # Seed from the FULL candidate pool, not from pa(y). pa(y) is sufficient in the
    # graph but the suite forbids conditioning on sessions (ID-3: IS is a collider of
    # arm and latent ability), and once IS is struck out the remaining parents can be
    # insufficient — e.g. for WR_1 -> LS_2 the backdoor WR_1 <- A_1 -> IS_1 -> LS_2
    # needs A_1, which is NOT a parent of LS_2. Seeding from the pool lets the prune
    # reach such nodes; seeding from pa(y) minus IS wrongly reports "no valid set".
    pool = set(candidate_adjusters(g, x, y))
    if not is_valid(g, x, y, pool):
        return []
    found: list[frozenset[str]] = []
    for _ in range(n_orders):
        order = list(pool)
        rng.shuffle(order)
        z = set(pool)
        for node in order:
            trial = z - {node}
            if is_valid(g, x, y, trial):
                z = trial
        fz = frozenset(z)
        if fz not in found:
            found.append(fz)
    return sorted(found, key=len)


def report(g: nx.DiGraph, x: str, y: str, title: str) -> list[frozenset[str]]:
    print(f"\n--- {title}: {x} -> {y} ---")
    pool = candidate_adjusters(g, x, y)
    parents = {p for p in g.predecessors(y)} - {x} - LATENT
    print(f"  candidate adjusters (measured, non-descendant, sessions excluded): {len(pool)}")
    check(g, x, y, parents, "(pa(y): the graph-theoretic answer, may include sessions)")
    if any(p.startswith("IS_") for p in parents):
        # Whether the suite's ID-3 ban on conditioning on sessions actually bites for
        # this coupling: does dropping IS from pa(y) break it, and can other measured
        # nodes repair it? This is the distinction that makes a coupling fittable
        # under suite conventions rather than merely identifiable on paper.
        check(g, x, y, parents - {p for p in parents if p.startswith("IS_")},
              "(pa(y) MINUS sessions — ID-3 forbids conditioning on IS)")
    check(g, x, y, set(pool), "(full candidate pool)")
    sets = minimal_sets(g, x, y)
    if not sets:
        print("  NO sufficient set exists within the allowed pool "
              "-> NOT identifiable without conditioning on sessions (ID-3 collider)")
        return sets
    for z in sets:
        measured = {n.split("_")[0] for n in z} - {"A", "HS", "IG"}
        needs_unmeasured = measured & UNMEASURED
        verdict = (
            "FITTABLE" if len(z) <= FITTABLE_MAX_ADJUSTERS and not needs_unmeasured
            else "too wide" if len(z) > FITTABLE_MAX_ADJUSTERS
            else f"needs unregistered measure(s) {sorted(needs_unmeasured)}"
        )
        print(f"  minimal set (|Z|={len(z)}): {sorted(z)}  -> {verdict}")
    return sets


def summarise(g: nx.DiGraph, rows: list[dict], out: Path) -> None:
    """One row per (coupling, transition) with the verdict the note tabulates."""
    frame = [
        {
            "coupling": r["title"],
            "exposure": r["x"],
            "outcome": r["y"],
            "transition": r["transition"],
            "direct_edge_in_dag": r["direct"],
            "min_adjusters_found": min((len(z) for z in r["sets"]), default=None),
            "minimal_set": (
                " + ".join(sorted(min(r["sets"], key=len))) if r["sets"] else ""
            ),
            "fittable": bool(
                r["sets"] and min(len(z) for z in r["sets"]) <= FITTABLE_MAX_ADJUSTERS
            ),
        }
        for r in rows
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    header = list(frame[0])
    lines = [",".join(header)]
    for row in frame:
        lines.append(",".join(f'"{row[h]}"' for h in header))
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("output/notes/202607241600-wr-bands"))
    args = parser.parse_args()

    template = parse_dagitty(DAG_PATH)
    assert nx.is_directed_acyclic_graph(template)
    assert_reverse_edges_match_dag(template)

    g = three_slice_unroll()
    assert nx.is_directed_acyclic_graph(g)
    print(f"3-slice unroll: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    targets = [
        ("LS", "reading -> letter sounds"),
        ("PA", "reading -> blending (phoneme awareness)"),
        ("NW", "reading -> nonword reading"),
    ]
    rows: list[dict] = []

    print("\n" + "=" * 70)
    print("TRANSITION 1 (t1->t2): randomised window, immediate arm only.")
    print("The exposure WR_1 is PRE-randomisation, so IG is not a confounder here.")
    print("=" * 70)
    for sym, title in targets:
        sets = report(g, "WR_1", f"{sym}_2", title)
        rows.append({"title": title, "x": "WR_1", "y": f"{sym}_2", "transition": 1,
                     "direct": sym in REVERSE, "sets": sets})

    print("\n" + "=" * 70)
    print("TRANSITION 2 (t2->t3): post-crossover, BOTH arms on programme.")
    print("IG must enter or the coupling absorbs a spurious crossover component.")
    print("=" * 70)
    for sym, title in targets:
        sets = report(g, "WR_2", f"{sym}_3", title)
        rows.append({"title": title, "x": "WR_2", "y": f"{sym}_3", "transition": 2,
                     "direct": sym in REVERSE, "sets": sets})

    print("\n" + "=" * 70)
    print("IS THERE A DIRECT WR -> NW EDGE AT ALL?")
    print("=" * 70)
    direct = ("NW" in REVERSE)
    print(f"  WR_t -> NW_t1 in the DAG: {direct}")
    print("  So a WR->NW coupling estimated on this graph is the TOTAL lagged effect")
    print("  routed through WR_t -> LS_t1 -> NW_t1 and WR_t -> PA_t1 -> NW_t1,")
    print("  not a direct edge. Adjusting for LS_t1 / PA_t1 would block it entirely:")
    check(g, "WR_1", "NW_2", {"NW_1", "LS_2", "PA_2", "A_1", "HS"},
          "(conditioning on the mediators kills the effect being sought)")

    print("\n" + "=" * 70)
    print("REGRESSION TEST: the design note's published results still hold")
    print("with LS added to the reverse-edge set (they were derived without it)")
    print("=" * 70)
    check(g, "WR_2", "TE_3", {"TE_2", "TR_2", "RW_2", "SP_2", "A_2", "HS"}, "(no IG -> expect NOT-VALID)")
    check(g, "WR_2", "TE_3", {"TE_2", "TR_2", "RW_2", "SP_2", "A_2", "HS", "IG"}, "(+ IG -> expect VALID)")
    check(g, "WR_2", "TR_3", {"TR_2", "RW_2", "A_2", "HS", "IG"}, "(expect VALID)")

    summarise(g, rows, args.out / "lagged_identifiability.csv")


if __name__ == "__main__":
    main()
