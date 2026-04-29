from __future__ import annotations

from math import comb
from typing import Dict, Sequence

import networkx as nx

from benchmark.eval.assertion_checker import evaluate_assertions, has_multi_net_pin, list_multi_net_pin_violations
from benchmark.eval.connection_metrics import compute_best_connection_rp
from benchmark.eval.erc_checker import check_ab_erc
from benchmark.eval.graph_utils import deserialize_graph
from tools.build_graph import validate_graph_structure


CONNECTION_THRESHOLD = 0.95
ASSERTION_PASS_THRESHOLD = 0.90


def check_structural_invariants(G: nx.Graph) -> bool:
    """
    Pass condition 1: graph structure invariants.
    """
    errors = validate_graph_structure(G)
    return len(errors) == 0


def check_critical_pins(G: nx.Graph) -> bool:
    """
    Pass condition 3: critical pins (PowerIn / Ground) must not be floating.
    """
    critical_types = {"PowerIn", "Ground"}

    for node, attr in G.nodes(data=True):
        if attr.get("type") != "Pin":
            continue

        if str(attr.get("electrical_type", "")) not in critical_types:
            continue

        has_net = any(G.nodes[n].get("type") == "Net" for n in G.neighbors(node))
        if not has_net:
            return False

    return True


def check_pass_single_with_graph(
    gt: Dict[str, object],
    G: nx.Graph,
    normalized_gen_nets: Sequence[Dict[str, object]],
    connection_threshold: float = CONNECTION_THRESHOLD,
) -> bool:
    """
    Single-run pass decision with already-deserialized graph and normalized nets.
    """
    if not check_structural_invariants(G):
        return False

    if has_multi_net_pin(G):
        return False

    if not check_critical_pins(G):
        return False

    assertions = gt.get("verification_assertions", [])
    if isinstance(assertions, list) and assertions:
        summary = evaluate_assertions(G, assertions)
        benchmark_total = int(summary.get("total", 0))
        benchmark_passed = int(summary.get("passed", 0))
        multi_net_pin_count = len(list_multi_net_pin_violations(G))

        denom = benchmark_total + multi_net_pin_count
        adjusted_assertion_pass_rate = (benchmark_passed / denom) if denom > 0 else 1.0
        return adjusted_assertion_pass_rate >= ASSERTION_PASS_THRESHOLD

    # Backward-compatible fallback: when design-level assertions are not provided,
    # keep legacy pass criterion to avoid breaking historical benchmark files.
    if not check_ab_erc(G, gt.get("erc_rule_instances", [])):
        return False

    recall, precision = compute_best_connection_rp(gt, normalized_gen_nets)
    if min(recall, precision) < connection_threshold:
        return False

    return True


def check_pass_single(
    gt: Dict[str, object],
    result: Dict[str, object],
    normalized_gen_nets: Sequence[Dict[str, object]],
    connection_threshold: float = CONNECTION_THRESHOLD,
) -> bool:
    """
    Backward-compatible wrapper that reads graph from one run result.
    """
    graph_data = result.get("graph_snapshot")
    if not isinstance(graph_data, dict):
        return False

    G = deserialize_graph(graph_data)
    return check_pass_single_with_graph(
        gt=gt,
        G=G,
        normalized_gen_nets=normalized_gen_nets,
        connection_threshold=connection_threshold,
    )


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator from OpenAI HumanEval protocol.
    """
    if n <= 0:
        return 0.0
    if c < 0:
        c = 0
    if c > n:
        c = n

    if k <= 0:
        return 0.0

    if n - c < k:
        return 1.0

    return 1.0 - comb(n - c, k) / comb(n, k)
