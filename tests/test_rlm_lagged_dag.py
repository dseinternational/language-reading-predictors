# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Structural guard for the adopted Byrne/RLM lagged DAG (#338/#409)."""

from __future__ import annotations

import re
from pathlib import Path

import networkx as nx

from language_reading_predictors.statistical_models.datasets import RLM_MEASURES

REPO = Path(__file__).resolve().parents[1]
BASE_PATH = REPO / "dag" / "dag-reading-language-memory.dagitty"
LAGGED_PATH = REPO / "dag" / "dag-reading-language-memory-lagged.dagitty"

OBSERVED = {
    "bassim",
    "basmat",
    "basdig",
    "basnum",
    "bpvs",
    "trog",
    "basread",
    "basspel",
    "woco",
}
STATIC = {"GA", "HS", "readgrp"}
REVERSE_TARGETS = {"bpvs", "trog", "basdig"}


def parse_dagitty(path: Path) -> tuple[nx.DiGraph, set[frozenset[str]]]:
    """Parse the repository's simple edge syntax and retain bidirected pairs."""
    src = path.read_text()
    body = src[src.index("dag {") + len("dag {") : src.rindex("}")]
    graph = nx.DiGraph()
    bidirected: set[frozenset[str]] = set()
    pending = ""
    for raw in body.splitlines():
        line = raw.split("//")[0].strip()
        if not line:
            continue
        if pending:
            line = f"{pending} {line}"
            pending = ""
        if "->" in line and "{" in line and "}" not in line:
            pending = line
            continue
        if "<->" in line:
            left, right = (part.strip() for part in line.split("<->"))
            bidirected.add(frozenset((left, right)))
            graph.add_nodes_from((left, right))
            continue
        if "->" not in line:
            match = re.match(r"^(\S+)\s+\[", line)
            if match:
                graph.add_node(match.group(1))
            continue
        match = re.match(r"^(\S+)\s*->\s*\{([^}]*)\}$", line)
        if match:
            graph.add_edges_from(
                (match.group(1), target) for target in match.group(2).split()
            )
            continue
        match = re.match(r"^(\S+)\s*->\s*(\S+)$", line)
        if match:
            graph.add_edge(match.group(1), match.group(2))
            continue
        raise ValueError(f"unparsed dagitty line: {line!r}")
    if pending:
        raise ValueError(f"unterminated dagitty edge: {pending!r}")
    return graph, bidirected


def rename_base(node: str, suffix: str) -> str:
    if node in STATIC:
        return node
    if node == "age":
        return f"age_{suffix}"
    return f"{node}_{suffix}"


def test_rlm_lagged_template_parses_and_is_acyclic():
    graph, bidirected = parse_dagitty(LAGGED_PATH)

    assert nx.is_directed_acyclic_graph(graph)
    assert graph.number_of_nodes() == 23
    assert graph.number_of_edges() == 97
    assert bidirected == {frozenset(("age_t", "readgrp"))}


def test_each_lagged_slice_copies_the_adopted_contemporaneous_graph():
    base, base_bidirected = parse_dagitty(BASE_PATH)
    lagged, _ = parse_dagitty(LAGGED_PATH)

    assert base_bidirected == {frozenset(("age", "readgrp"))}
    for suffix in ("t", "t1"):
        expected = {
            (rename_base(source, suffix), rename_base(target, suffix))
            for source, target in base.edges
        }
        slice_nodes = STATIC | {f"age_{suffix}"} | {
            f"{symbol}_{suffix}" for symbol in OBSERVED
        }
        actual = {
            (source, target)
            for source, target in lagged.edges
            if source in slice_nodes and target in slice_nodes
        }
        assert actual == expected


def test_carryover_and_reverse_edges_are_exactly_pre_specified():
    graph, _ = parse_dagitty(LAGGED_PATH)

    cross_slice = {
        (source, target)
        for source, target in graph.edges
        if source.endswith("_t") and target.endswith("_t1")
    }
    carryover = {
        (f"{symbol}_t", f"{symbol}_t1") for symbol in OBSERVED | {"age"}
    }
    reverse = {
        ("basread_t", f"{target}_t1") for target in REVERSE_TARGETS
    }
    assert cross_slice == carryover | reverse


def test_reverse_targets_match_source_scope_and_have_confirmed_bounds():
    assert REVERSE_TARGETS == {"bpvs", "trog", "basdig"}
    assert all(RLM_MEASURES[symbol].n_trials_confirmed for symbol in REVERSE_TARGETS)
    assert not (
        REVERSE_TARGETS
        & {"basspel", "woco", "basnum", "bassim", "basmat"}
    )


def test_reverse_couplings_retain_unmeasured_common_causes():
    graph, _ = parse_dagitty(LAGGED_PATH)

    for target in REVERSE_TARGETS:
        later = f"{target}_t1"
        assert {"GA", "HS"} <= set(graph.predecessors("basread_t"))
        assert {"GA", "HS"} <= set(graph.predecessors(later))

        backdoor = graph.copy()
        backdoor.remove_edges_from(list(backdoor.out_edges("basread_t")))
        assert nx.has_path(backdoor.to_undirected(), "basread_t", later)
