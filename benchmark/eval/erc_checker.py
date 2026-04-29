from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

from tools.build_graph import get_component_data
from tools.erc_verification import ERCIssue, ERCVerificationEngine


_TARGET_PIN_PARAM_KEYS: Tuple[str, ...] = (
    "target_pins",
    "target_pin",
    "timing_pins",
    "gain_pin_1",
    "gain_pin_2",
    "rg_pin_1",
    "rg_pin_2",
    "in_minus_pin",
    "in_plus_pin",
    "out_pin",
    "sense_pins",
    "fb_pin",
    "output_pins",
    "pin_1",
    "pin_2",
    "pin_in",
    "pin_out",
    "xtal_in",
    "xtal_out",
    "positive_pin",
    "negative_pin",
    "analog_gnd_pins",
    "digital_gnd_pins",
)


def _norm_pin_token(token: str) -> str:
    return str(token).strip().upper()


def _base_pin_token(token: str) -> str:
    return re.sub(r"_\d+$", "", _norm_pin_token(token))


def _pin_token_match(a: str, b: str) -> bool:
    na = _norm_pin_token(a)
    nb = _norm_pin_token(b)
    if na == nb:
        return True
    return _base_pin_token(na) == _base_pin_token(nb)


def _parse_target(target: str) -> Tuple[Optional[str], Optional[str]]:
    target = str(target).strip()
    if not target:
        return None, None

    if "." not in target:
        return target, None

    comp, pin = target.split(".", 1)
    return comp.strip() or None, pin.strip() or None


def _extract_target_pins_from_rule(rule: Dict[str, object]) -> List[str]:
    params = rule.get("parameters", {})
    if not isinstance(params, dict):
        return []

    values: List[str] = []
    for key in _TARGET_PIN_PARAM_KEYS:
        val = params.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            values.extend(str(x).strip() for x in val if str(x).strip())
        else:
            tok = str(val).strip()
            if tok:
                values.append(tok)

    return values


def lookup_rule_for_instance(G: nx.Graph, inst: Dict[str, object]) -> Optional[Dict[str, object]]:
    """
    Resolve one compact GT rule instance to a full library rule dict.
    """
    rule_type = str(inst.get("rule", "")).strip()
    comp_id, target_pin = _parse_target(str(inst.get("target", "")))

    if not rule_type or not comp_id or not G.has_node(comp_id):
        return None

    part_number = str(G.nodes[comp_id].get("part_number", "")).strip()
    lib_data = get_component_data(part_number)
    if not lib_data:
        return None

    candidates = lib_data.get("erc_rules", [])
    for rule in candidates:
        if str(rule.get("rule_type", "")).strip() != rule_type:
            continue

        if not target_pin:
            return rule

        target_pins = _extract_target_pins_from_rule(rule)
        if not target_pins:
            return rule

        if any(_pin_token_match(target_pin, p) for p in target_pins):
            return rule

    return None


def evaluate_erc_instances(
    G: nx.Graph,
    erc_instances: Sequence[Dict[str, object]],
    classes: Iterable[str] = ("A", "B"),
    stop_on_first_failure: bool = False,
) -> Dict[str, object]:
    """
    Evaluate GT ERC instances on one generated graph.
    """
    class_filter: Set[str] = {str(c).upper() for c in classes}
    engine = ERCVerificationEngine(G)

    total = 0
    checked = 0
    passed = 0
    failed = 0
    unresolved = 0
    violations: List[Dict[str, object]] = []

    for inst in erc_instances or []:
        inst_class = str(inst.get("class", "")).upper()
        if class_filter and inst_class not in class_filter:
            continue

        total += 1
        comp_id, _ = _parse_target(str(inst.get("target", "")))
        if not comp_id:
            unresolved += 1
            continue

        rule = lookup_rule_for_instance(G, inst)
        if rule is None:
            unresolved += 1
            continue

        checked += 1
        issues: List[ERCIssue] = engine.verify_rule(comp_id, rule)
        hard_issues = [issue for issue in issues if issue.get("severity") == "error"]

        if hard_issues:
            failed += 1
            violations.append(
                {
                    "instance": inst,
                    "issues": hard_issues,
                }
            )
            if stop_on_first_failure:
                break
        else:
            passed += 1

    # Strict pass_rate treats unresolved instances as failures to avoid optimistic bias.
    pass_rate = passed / total if total > 0 else 1.0
    pass_rate_checked = passed / checked if checked > 0 else 1.0
    return {
        "total": total,
        "checked": checked,
        "passed": passed,
        "failed": failed,
        "unresolved": unresolved,
        "pass_rate": pass_rate,
        "pass_rate_checked": pass_rate_checked,
        "violations": violations,
    }


def check_ab_erc(
    G: nx.Graph,
    erc_instances: Sequence[Dict[str, object]],
) -> bool:
    """
    Condition 2 in pass@k: all class A+B checked rules must pass.
    """
    summary = evaluate_erc_instances(G, erc_instances, classes=("A", "B"), stop_on_first_failure=True)
    total = int(summary.get("total", 0))
    checked = int(summary.get("checked", 0))
    failed = int(summary.get("failed", 0))
    unresolved = int(summary.get("unresolved", 0))

    # unresolved means the rule instance could not be mapped/evaluated -> treat as non-pass.
    if failed > 0 or unresolved > 0:
        return False

    if total > 0 and checked != total:
        return False

    return True
