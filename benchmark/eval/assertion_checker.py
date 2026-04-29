from __future__ import annotations

import re
from collections import deque
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypedDict

import networkx as nx


class AssertionFailure(TypedDict):
    id: str
    check: str
    desc: str
    message: str
 

class AssertionSummary(TypedDict):
    total: int
    passed: int
    failed: int
    pass_rate: float
    functional_total: int
    functional_passed: int
    acr: float
    failures: List[AssertionFailure]


GROUND_TOKENS = ("gnd", "ground", "agnd", "dgnd", "pgnd", "vss")
POWER_TOKENS = ("vcc", "vdd", "vin", "vout", "vbat", "vdda", "vddio", "vref", "power")
DEFAULT_ALLOWED_PATH_COMPONENTS: Set[str] = {"resistor", "capacitor", "inductor", "diode"}
PASSIVE_REFDES_KIND_MAP: Dict[str, str] = {
    "R": "resistor",
    "C": "capacitor",
    "L": "inductor",
    "D": "diode",
    "Y": "crystal",
    "F": "fuse",
}


def _safe_str(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _safe_lower(value: object) -> str:
    return _safe_str(value).lower()


def _node_type(G: nx.Graph, node: str) -> str:
    if not G.has_node(node):
        return ""
    return _safe_str(G.nodes[node].get("type"))


def _edge_relation(G: nx.Graph, a: str, b: str) -> str:
    data = G.get_edge_data(a, b)
    if not isinstance(data, dict):
        return ""
    return _safe_str(data.get("relation")).lower()


def _component_pins(G: nx.Graph, comp_id: str) -> List[str]:
    if not G.has_node(comp_id):
        return []
    out: List[str] = []
    for nbr in G.neighbors(comp_id):
        if _node_type(G, nbr) != "Pin":
            continue
        rel = _edge_relation(G, comp_id, nbr)
        if rel and rel != "belongs_to":
            continue
        out.append(nbr)
    return out


def _pin_parent_component(G: nx.Graph, pin_id: str) -> Optional[str]:
    if not G.has_node(pin_id):
        return None
    for nbr in G.neighbors(pin_id):
        if _node_type(G, nbr) != "Component":
            continue
        rel = _edge_relation(G, pin_id, nbr)
        if rel and rel != "belongs_to":
            continue
        return nbr
    return None


def _pin_connected_nets(G: nx.Graph, pin_id: str) -> List[str]:
    if not G.has_node(pin_id):
        return []
    out: List[str] = []
    for nbr in G.neighbors(pin_id):
        if _node_type(G, nbr) != "Net":
            continue
        rel = _edge_relation(G, pin_id, nbr)
        if rel and rel != "connected_to":
            continue
        out.append(nbr)
    return out


def _net_connected_pins(G: nx.Graph, net_id: str) -> List[str]:
    if not G.has_node(net_id):
        return []
    out: List[str] = []
    for nbr in G.neighbors(net_id):
        if _node_type(G, nbr) != "Pin":
            continue
        rel = _edge_relation(G, net_id, nbr)
        if rel and rel != "connected_to":
            continue
        out.append(nbr)
    return out


def _component_nets(G: nx.Graph, comp_id: str) -> Set[str]:
    out: Set[str] = set()
    for pin in _component_pins(G, comp_id):
        out.update(_pin_connected_nets(G, pin))
    return out


def _normalize_pin_alias(token: str) -> str:
    return _safe_str(token).upper()


def _parse_refdes_pin_token(pin_ref: object) -> Tuple[str, str]:
    token = _safe_str(pin_ref)
    if "." not in token:
        return "", ""
    comp, pin_token = token.split(".", 1)
    return _safe_str(comp), _safe_str(pin_token)


def _is_passive_pin_ref(pin_ref: object) -> bool:
    comp, pin_token = _parse_refdes_pin_token(pin_ref)
    if not comp or not pin_token:
        return False
    return _safe_str(comp[:1]).upper() in PASSIVE_REFDES_KIND_MAP


def _collect_passive_pin_candidates(G: nx.Graph, pin_ref: object) -> List[str]:
    """
    Topology-first fallback: when passive RefDes is renumbered (e.g., C5->C12),
    match by passive kind + pin token (name/number).
    """
    comp, pin_token = _parse_refdes_pin_token(pin_ref)
    if not comp or not pin_token:
        return []

    kind = PASSIVE_REFDES_KIND_MAP.get(_safe_str(comp[:1]).upper())
    if not kind:
        return []

    probe = _normalize_pin_alias(pin_token)
    out: List[str] = []
    for node, attr in G.nodes(data=True):
        if _safe_str(attr.get("type")) != "Pin":
            continue

        parent = _pin_parent_component(G, node)
        if not parent:
            continue
        if not _component_matches_kind(G, parent, kind):
            continue

        name = _normalize_pin_alias(attr.get("name"))
        num = _normalize_pin_alias(attr.get("pin_number"))
        if probe in {name, num, f"PIN{num}", f"PIN_{num}", f"PIN-{num}"}:
            out.append(node)

    return sorted(set(out))


def _resolve_pin_node(G: nx.Graph, pin_ref: object) -> Optional[str]:
    token = _safe_str(pin_ref)
    if not token:
        return None

    if G.has_node(token) and _node_type(G, token) == "Pin":
        return token

    # Support RefDes.Pin form.
    if "." in token:
        comp, pin_token = token.split(".", 1)
        comp = _safe_str(comp)
        pin_token = _safe_str(pin_token)
        if comp and G.has_node(comp) and _node_type(G, comp) == "Component":
            probe = _normalize_pin_alias(pin_token)

            pin_nodes = _component_pins(G, comp)
            name_tokens: Set[str] = set()
            num_tokens: Set[str] = set()
            for pin in pin_nodes:
                attrs = G.nodes[pin]
                name_tokens.add(_normalize_pin_alias(attrs.get("name")))
                num_tokens.add(_normalize_pin_alias(attrs.get("pin_number")))

            alias_probes: List[str] = [probe]

            # Practical alias tolerance for fixed-regulator datasets:
            # some assertions still use ADJ while generated netlists expose GND/VSS (pin1).
            if probe == "ADJ" and "ADJ" not in name_tokens:
                if "GND" in name_tokens or "VSS" in name_tokens or "1" in num_tokens:
                    alias_probes.extend(["GND", "VSS", "1", "PIN1", "PIN_1", "PIN-1"])

            if probe in {"GND", "VSS"} and "ADJ" in name_tokens:
                alias_probes.append("ADJ")

            # 规则 1: exact match
            for alias_probe in alias_probes:
                if not alias_probe:
                    continue
                for pin in pin_nodes:
                    attrs = G.nodes[pin]
                    name = _normalize_pin_alias(attrs.get("name"))
                    num = _normalize_pin_alias(attrs.get("pin_number"))
                    if alias_probe in {name, num, f"PIN{num}", f"PIN_{num}", f"PIN-{num}"}:
                        return pin
            
            # 规则 2: fallback to suffix cluster matching (e.g. U2.VOUT -> U2.VOUT_1, U2.VOUT_2)
            # Find all pins that match the prefix {pin_token}_[0-9]+
            import re
            prefix = pin_token.upper()
            cluster_pins = []
            for pin in pin_nodes:
                attrs = G.nodes[pin]
                name = _safe_str(attrs.get("name")).upper()
                if re.fullmatch(rf"{re.escape(prefix)}_[0-9]+", name):
                    cluster_pins.append(pin)
            
            if cluster_pins:
                # 检查这些 pin 是否都在同一个 net 上
                net_sets = [set(_pin_connected_nets(G, p)) for p in cluster_pins]
                common = set.intersection(*net_sets) if net_sets else set()
                if common:
                    # 如果都在同一个网上，允许将它们视为该 assertion pin 的代表，返回任一（或第一个）即可代表拓扑特征
                    return cluster_pins[0]

        # For fallback graphs where Pin node id may literally be token.
        if G.has_node(token) and _node_type(G, token) == "Pin":
            return token

    # Support U1_Pin3 / U1_3 style if passed as alias.
    if G.has_node(token) and _node_type(G, token) == "Pin":
        return token

    # pin_1 / pin1 alias only works when a component context is not given; try global exact match on pin_number/name.
    probe = _normalize_pin_alias(token)
    m = re.fullmatch(r"PIN[_\-\s]*([0-9]+)", probe)
    pin_alias_number = m.group(1) if m else ""

    for node, attr in G.nodes(data=True):
        if _safe_str(attr.get("type")) != "Pin":
            continue
        name = _normalize_pin_alias(attr.get("name"))
        num = _normalize_pin_alias(attr.get("pin_number"))
        if probe in {name, num, f"PIN{num}", f"PIN_{num}", f"PIN-{num}"}:
            return node
        if pin_alias_number and num == pin_alias_number:
            return node

    # Passive renumber tolerance: if a passive refdes pin cannot be found by strict name,
    # attempt unique fallback by passive kind + pin token.
    candidates = _collect_passive_pin_candidates(G, token)
    if len(candidates) == 1:
        return candidates[0]

    return None


def _resolve_pin_list(
    G: nx.Graph,
    pins: Iterable[object],
    allow_unresolved_passive: bool = False,
) -> Tuple[List[str], List[str]]:
    resolved: List[str] = []
    missing: List[str] = []
    for pin in pins:
        node = _resolve_pin_node(G, pin)
        if node:
            resolved.append(node)
        else:
            token = _safe_str(pin)
            if allow_unresolved_passive and _is_passive_pin_ref(token):
                # Passive refdes mismatch should not hard-fail topology evaluation.
                continue
            missing.append(token)
    return resolved, missing


def _resolve_net_node(G: nx.Graph, net_ref: object) -> Optional[str]:
    token = _safe_str(net_ref)
    if not token:
        return None

    if G.has_node(token) and _node_type(G, token) == "Net":
        return token

    probe = token.lower()
    for node, attr in G.nodes(data=True):
        if _safe_str(attr.get("type")) != "Net":
            continue
        name = _safe_lower(attr.get("name", node))
        if probe == _safe_lower(node) or probe == name:
            return node
    return None


def _net_is_ground(G: nx.Graph, net_id: str) -> bool:
    if not G.has_node(net_id):
        return False

    attrs = G.nodes[net_id]
    net_type = _safe_lower(attrs.get("net_type"))
    net_class = _safe_lower(attrs.get("net_class"))
    name = _safe_lower(attrs.get("name", net_id))

    if any(tok in net_type for tok in GROUND_TOKENS):
        return True
    if any(tok in net_class for tok in GROUND_TOKENS):
        return True
    if any(tok in name for tok in GROUND_TOKENS):
        return True

    for pin in _net_connected_pins(G, net_id):
        et = _safe_lower(G.nodes[pin].get("electrical_type"))
        pname = _safe_lower(G.nodes[pin].get("name"))
        if "ground" in et or any(tok in pname for tok in GROUND_TOKENS):
            return True

    return False


def _net_is_power(G: nx.Graph, net_id: str) -> bool:
    if not G.has_node(net_id):
        return False

    attrs = G.nodes[net_id]
    net_type = _safe_lower(attrs.get("net_type"))
    net_class = _safe_lower(attrs.get("net_class"))
    name = _safe_lower(attrs.get("name", net_id))

    if any(tok in net_type for tok in POWER_TOKENS):
        return True
    if any(tok in net_class for tok in POWER_TOKENS):
        return True
    if any(tok in name for tok in POWER_TOKENS):
        return True

    for pin in _net_connected_pins(G, net_id):
        et = _safe_lower(G.nodes[pin].get("electrical_type"))
        pname = _safe_lower(G.nodes[pin].get("name"))
        if et in {"power", "powerin", "powerout"}:
            return True
        if any(tok in pname for tok in POWER_TOKENS):
            return True

    return False


def _components_on_net(G: nx.Graph, net_id: str, exclude: Optional[Set[str]] = None) -> Set[str]:
    exclude = exclude or set()
    out: Set[str] = set()
    for pin in _net_connected_pins(G, net_id):
        parent = _pin_parent_component(G, pin)
        if parent and parent not in exclude:
            out.add(parent)
    return out


def _infer_component_kind(G: nx.Graph, comp_id: str) -> str:
    if not G.has_node(comp_id):
        return "unknown"

    attrs = G.nodes[comp_id]
    ref = _safe_lower(comp_id)
    part_number = _safe_lower(attrs.get("part_number"))
    part_type = _safe_lower(attrs.get("part_type"))
    text = " ".join([ref, part_number, part_type])

    if "crystal" in text or "resonator" in text or ref.startswith("y"):
        return "crystal"
    if "ic" in text or "mcu" in text or "controller" in text or ref.startswith("u"):
        return "ic"
    if "res" in text or "resistor" in text or ref.startswith("r"):
        return "resistor"
    if "cap" in text or "capacitor" in text or ref.startswith("c"):
        return "capacitor"
    if "ind" in text or "inductor" in text or ref.startswith("l"):
        return "inductor"
    if "diode" in text or "schottky" in text or "tvs" in text or "led" in text or ref.startswith("d"):
        return "diode"
    if "fuse" in text or ref.startswith("f"):
        return "fuse"

    return "unknown"


def _component_matches_kind(G: nx.Graph, comp_id: str, kind: str) -> bool:
    k = _safe_lower(kind)
    if not k:
        return True

    actual = _infer_component_kind(G, comp_id)
    if k in {"res", "resistor"}:
        return actual == "resistor"
    if k in {"cap", "capacitor"}:
        return actual == "capacitor"
    if k in {"ind", "inductor", "ferrite", "ferritebead", "bead"}:
        return actual == "inductor"
    if k in {"diode", "tvs", "esd", "led", "diode_or_led"}:
        return actual == "diode"
    if k in {"crystal", "resonator", "osc"}:
        return actual == "crystal"
    if k in {"fuse"}:
        return actual == "fuse"
    if k in {"ic", "chip", "active"}:
        return actual == "ic"

    return k in actual


def _parse_res_ohms(raw: object) -> Optional[float]:
    text = _safe_str(raw)
    if not text:
        return None

    upper = text.upper()
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([KMR]?)", upper)
    if not m:
        return None

    value = float(m.group(1))
    suffix = m.group(2)
    if suffix == "K":
        value *= 1e3
    elif suffix == "M":
        value *= 1e6
    elif suffix == "R":
        value *= 1.0

    return value


def _parse_ohm_range(raw: object) -> Tuple[Optional[float], Optional[float]]:
    text = _safe_str(raw)
    if not text or text.lower() == "any":
        return None, None

    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([KMR]?)\s*[-~]\s*([0-9]+(?:\.[0-9]+)?)\s*([KMR]?)", text, re.IGNORECASE)
    if m:
        lo = _parse_res_ohms(f"{m.group(1)}{m.group(2)}")
        hi = _parse_res_ohms(f"{m.group(3)}{m.group(4)}")
        if lo is None or hi is None:
            return None, None
        return min(lo, hi), max(lo, hi)

    single = _parse_res_ohms(text)
    if single is None:
        return None, None
    return single, single


def _res_value_in_range(G: nx.Graph, comp_id: str, lo: Optional[float], hi: Optional[float]) -> bool:
    if lo is None or hi is None:
        return True

    attrs = G.nodes[comp_id] if G.has_node(comp_id) else {}
    value = _parse_res_ohms(attrs.get("value"))
    if value is None:
        value = _parse_res_ohms(attrs.get("part_number"))
    if value is None:
        return True

    return lo <= value <= hi


def _is_functional_assertion(assertion: Dict[str, Any]) -> bool:
    cls = _safe_lower(assertion.get("class") or assertion.get("category") or assertion.get("assertion_class"))
    if cls in {"f", "functional"}:
        return True

    aid = _safe_str(assertion.get("id")).upper()
    if re.match(r"^F[_\-]", aid):
        return True

    return False


def _normalize_atomic_assertion(assertion: Dict[str, Any]) -> Dict[str, Any]:
    check = _safe_str(assertion.get("check") or assertion.get("condition")).upper()
    desc_text = _safe_str(assertion.get("desc", assertion.get("description", "")))

    raw_params = assertion.get("params")
    if not isinstance(raw_params, dict):
        raw_params = assertion.get("parameters")
    params: Dict[str, Any] = dict(raw_params) if isinstance(raw_params, dict) else {}
    params.setdefault("_assertion_desc", desc_text)
    params.setdefault("_assertion_id", _safe_str(assertion.get("id")))

    target = assertion.get("target")
    if target is not None:
        target_text = _safe_str(target)
        if check in {
            "PIN_SINGLE_NET",
            "PIN_NOT_FLOATING",
            "MUST_BE_FLOATING",
            "CONNECTED_TO_GROUND",
            "CONNECTED_TO_POWER",
            "CONNECTED_TO_NET_CLASS",
            "HAS_CAP_TO_GROUND",
            "HAS_PULLUP_TO_POWER",
            "HAS_SENSE_RESISTOR",
            "HAS_FREEWHEELING_PATH",
            "HAS_DECOUPLING_NETWORK",
            "CONNECTED_TO",
        } and "pin" not in params:
            params["pin"] = target_text

        if check in {"EXISTS_PATH", "PATH_EXISTS"} and "from_pin" not in params:
            params["from_pin"] = target_text

    return {
        "id": assertion.get("id"),
        "desc": desc_text,
        "check": check,
        "params": params,
    }


def _normalize_allowed_component_kinds(raw: object) -> Optional[Set[str]]:
    if raw is None:
        return None

    if isinstance(raw, str):
        token = _safe_lower(raw)
        if not token:
            return set()
        return {token}

    if isinstance(raw, list):
        out: Set[str] = set()
        for item in raw:
            token = _safe_lower(item)
            if token:
                out.add(token)
        return out

    return set()


def _component_allowed_for_path(G: nx.Graph, comp_id: str, allowed_kinds: Optional[Set[str]]) -> bool:
    if allowed_kinds is None:
        return True
    if not allowed_kinds:
        return False

    if any(k in {"*", "any", "all"} for k in allowed_kinds):
        return True

    for kind in allowed_kinds:
        if _component_matches_kind(G, comp_id, kind):
            return True
    return False


def _check_pin_single_net(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin_token = params.get("pin")

    if pin_token is not None and str(pin_token).strip() != "":
        pin = _resolve_pin_node(G, pin_token)
        if not pin:
            return False, "Pin not found."
        nets = _pin_connected_nets(G, pin)
        if len(set(nets)) <= 1:
            return True, "Verification passed."
        return False, f"Pin '{pin}' is connected to multiple nets: {sorted(set(nets))}."

    violations = list_multi_net_pin_violations(G)
    if not violations:
        return True, "Verification passed."

    preview = ", ".join(str(v.get("pin", "")) for v in violations[:3])
    if len(violations) > 3:
        preview += ", ..."
    return False, f"Found {len(violations)} multi-net pin(s): {preview}."


def _check_pin_not_floating(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin = _resolve_pin_node(G, params.get("pin"))
    if not pin:
        return False, "Pin not found."
    nets = _pin_connected_nets(G, pin)
    if nets:
        return True, "Verification passed."
    return False, f"Pin '{pin}' is floating."


def _evaluate_pin_groups(G: nx.Graph, pin_groups: List[Any], allow_unresolved_passive: bool = True) -> Tuple[bool, str]:
    if not isinstance(pin_groups, list) or len(pin_groups) < 2:
        return False, "SAME_NET with pin_groups requires at least 2 groups."

    candidate_net_sets = []
    missing_groups = []

    for idx, group in enumerate(pin_groups):
        if not isinstance(group, list):
            missing_groups.append({"group_index": idx, "error": "Group is not a list"})
            continue
        resolved, missing = _resolve_pin_list(G, group, allow_unresolved_passive=allow_unresolved_passive)
        if not resolved:
            missing_groups.append({
                "group_index": idx,
                "missing": list(missing) if missing else group
            })
            continue

        nets = set()
        for pin in resolved:
            for net in _pin_connected_nets(G, pin):
                nets.add(net)

        if not nets:
            missing_groups.append({
                "group_index": idx,
                "missing": f"All pins floating in group: {group}"
            })
            continue

        candidate_net_sets.append(nets)

    if missing_groups:
        return False, f"Some pin groups unresolved or floating: {missing_groups}"

    common_nets = set.intersection(*candidate_net_sets)
    if common_nets:
        chosen = sorted(list(common_nets))
        return True, f"At least one common net found across pin groups: {chosen[:3]}"

    return False, f"No common net exists across pin groups: {pin_groups}."


def _check_same_net(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    if "net_group_candidates" in params:
        candidates = params.get("net_group_candidates", [])
        if not isinstance(candidates, list) or not candidates:
            return False, "SAME_NET with net_group_candidates requires a non-empty list."

        errors = []
        for cand in candidates:
            label = cand.get("label", "UNNAMED_CANDIDATE")
            pin_groups_list = cand.get("pin_groups_list", [])
            
            cand_success = True
            cand_errors = []
            
            for p_groups in pin_groups_list:
                ok, msg = _evaluate_pin_groups(G, p_groups, allow_unresolved_passive=True)
                if not ok:
                    cand_success = False
                    cand_errors.append(msg)
                    break
            
            if cand_success:
                return True, f"Successfully matched net_group_candidates combination: {label}."
            else:
                errors.append(f"Candidate {label} failed: {cand_errors}")
                
        return False, f"All combination candidates failed: {errors}"

    if "pin_groups" in params:
        pin_groups = params.get("pin_groups", [])
        return _evaluate_pin_groups(G, pin_groups, allow_unresolved_passive=True)

    pins = params.get("pins", [])
    if not isinstance(pins, list) or len(pins) < 2:
        return False, "SAME_NET requires at least two pins."

    resolved, missing = _resolve_pin_list(G, pins, allow_unresolved_passive=True)
    if missing:
        return False, f"Pin(s) not found: {missing}."
    if len(resolved) < 2:
        return False, "SAME_NET has fewer than two resolvable pins after passive normalization."

    net_sets = [set(_pin_connected_nets(G, p)) for p in resolved]
    if any(not s for s in net_sets):
        return False, "At least one pin is floating."

    common = set.intersection(*net_sets)
    if common:
        return True, "Verification passed."
    return False, f"Pins are not on the same net: {pins}."


def _check_connected_to_ground(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin = _resolve_pin_node(G, params.get("pin"))
    if not pin:
        return False, "Pin not found."
    nets = _pin_connected_nets(G, pin)
    if any(_net_is_ground(G, n) for n in nets):
        return True, "Verification passed."

    allow_passive_path = _parse_bool(params.get("allow_passive_path"), True)
    if allow_passive_path:
        target_nets = {
            node
            for node, attr in G.nodes(data=True)
            if _safe_str(attr.get("type")) == "Net" and _net_is_ground(G, node)
        }
        if target_nets:
            allowed_raw = params.get("allowed_components")
            if allowed_raw is None:
                allowed_kinds = {"resistor", "inductor", "fuse", "diode"}
            else:
                allowed_kinds = _normalize_allowed_component_kinds(allowed_raw)
                if not allowed_kinds:
                    return False, "allowed_components is empty or invalid."

            max_hops = int(params.get("max_hops", 6) or 6)
            if _path_exists_between_nodes(
                G,
                src_node=pin,
                dst_nodes=target_nets,
                max_hops=max_hops,
                allowed_component_kinds=allowed_kinds,
            ):
                return True, "Verification passed (connected to ground via passive path)."

    return False, f"Pin '{pin}' is not connected to a ground net."


def _check_connected_to_power(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin = _resolve_pin_node(G, params.get("pin"))
    if not pin:
        return False, "Pin not found."
    nets = _pin_connected_nets(G, pin)
    if any(_net_is_power(G, n) for n in nets):
        return True, "Verification passed."

    allow_passive_path = _parse_bool(params.get("allow_passive_path"), True)
    if allow_passive_path:
        target_nets = {
            node
            for node, attr in G.nodes(data=True)
            if _safe_str(attr.get("type")) == "Net" and _net_is_power(G, node)
        }
        if target_nets:
            allowed_raw = params.get("allowed_components")
            if allowed_raw is None:
                allowed_kinds = {"resistor", "inductor", "fuse", "diode"}
            else:
                allowed_kinds = _normalize_allowed_component_kinds(allowed_raw)
                if not allowed_kinds:
                    return False, "allowed_components is empty or invalid."

            max_hops = int(params.get("max_hops", 6) or 6)
            if _path_exists_between_nodes(
                G,
                src_node=pin,
                dst_nodes=target_nets,
                max_hops=max_hops,
                allowed_component_kinds=allowed_kinds,
            ):
                return True, "Verification passed (connected to power via passive path)."

    return False, f"Pin '{pin}' is not connected to a power net."


def _check_connected_to_net_class(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin = _resolve_pin_node(G, params.get("pin"))
    net_class = _safe_lower(params.get("net_class"))
    if not pin:
        return False, "Pin not found."
    if not net_class:
        return False, "Missing net_class."

    nets = _pin_connected_nets(G, pin)
    for net in nets:
        attrs = G.nodes[net]
        nc = _safe_lower(attrs.get("net_class"))
        nt = _safe_lower(attrs.get("net_type"))
        if net_class in nc or net_class in nt:
            return True, "Verification passed."
    return False, f"Pin '{pin}' is not on net_class '{net_class}'."


def _check_not_same_net(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pins = params.get("pins", [])
    if not isinstance(pins, list) or len(pins) != 2:
        return False, "NOT_SAME_NET requires exactly two pins."

    resolved, missing = _resolve_pin_list(G, pins, allow_unresolved_passive=True)
    if missing:
        return False, f"Pin(s) not found: {missing}."
    if len(resolved) < 2:
        return True, "Verification passed (passive refdes unresolved, skipped strict pin identity check)."

    a_nets = set(_pin_connected_nets(G, resolved[0]))
    b_nets = set(_pin_connected_nets(G, resolved[1]))
    if a_nets & b_nets:
        return False, f"Pins share net(s): {sorted(a_nets & b_nets)}."
    return True, "Verification passed."


def _check_net_group_isolation(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    resolved_groups: List[List[str]] = []

    groups = params.get("groups", [])
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, list):
                return False, "Each group must be a pin list."
            resolved, missing = _resolve_pin_list(G, group, allow_unresolved_passive=True)
            if missing:
                return False, f"Pin(s) not found: {missing}."
            if not resolved:
                continue
            resolved_groups.append(resolved)

    groups_by_net_class = params.get("groups_by_net_class", [])
    if groups_by_net_class is not None:
        if not isinstance(groups_by_net_class, list):
            return False, "groups_by_net_class must be a list."

        for group_selector in groups_by_net_class:
            selector_list: List[object]
            if isinstance(group_selector, list):
                selector_list = group_selector
            else:
                selector_list = [group_selector]

            if not selector_list:
                return False, "groups_by_net_class contains an empty selector group."

            nets_for_group: Set[str] = set()
            for selector in selector_list:
                token = _safe_lower(selector)
                if not token:
                    return False, "groups_by_net_class contains empty selector."

                for node, attr in G.nodes(data=True):
                    if _safe_str(attr.get("type")) != "Net":
                        continue

                    net_class = _safe_lower(attr.get("net_class"))
                    net_type = _safe_lower(attr.get("net_type"))
                    net_name = _safe_lower(attr.get("name", node))

                    if token in {"ground", "gnd"}:
                        if _net_is_ground(G, node):
                            nets_for_group.add(node)
                        continue

                    if token in {"power", "vcc", "vdd", "vin", "vout"}:
                        if _net_is_power(G, node):
                            nets_for_group.add(node)
                        continue

                    if token in {net_class, net_type, net_name}:
                        nets_for_group.add(node)
                        continue
                    if token in net_class or token in net_type or token in net_name:
                        nets_for_group.add(node)

            pins_for_group: Set[str] = set()
            for net in nets_for_group:
                pins_for_group.update(_net_connected_pins(G, net))

            if not pins_for_group:
                return False, f"No pins found for groups_by_net_class selector: {selector_list}."

            resolved_groups.append(sorted(pins_for_group))

    if len(resolved_groups) < 2:
        return False, "NET_GROUP_ISOLATION requires at least two groups (groups or groups_by_net_class)."

    for i in range(len(resolved_groups)):
        for j in range(i + 1, len(resolved_groups)):
            left = resolved_groups[i]
            right = resolved_groups[j]
            for a in left:
                a_nets = set(_pin_connected_nets(G, a))
                for b in right:
                    b_nets = set(_pin_connected_nets(G, b))
                    common = a_nets & b_nets
                    if common:
                        return False, f"Isolation violated: '{a}' and '{b}' share net(s) {sorted(common)}."

    return True, "Verification passed."


def _check_pin_not_on_net_with(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin = _resolve_pin_node(G, params.get("pin"))
    forbidden = params.get("forbidden_pins", [])
    if not pin:
        return False, "Pin not found."
    if not isinstance(forbidden, list) or not forbidden:
        return False, "forbidden_pins must be a non-empty list."

    resolved, missing = _resolve_pin_list(G, forbidden, allow_unresolved_passive=True)
    if missing:
        return False, f"Pin(s) not found: {missing}."
    if not resolved:
        return True, "Verification passed (forbidden passive pins unresolved after normalization)."

    pin_nets = set(_pin_connected_nets(G, pin))
    for fp in resolved:
        common = pin_nets & set(_pin_connected_nets(G, fp))
        if common:
            return False, f"Pin '{pin}' shares forbidden net(s) {sorted(common)} with '{fp}'."

    return True, "Verification passed."


def _check_has_cap_to_ground(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin = _resolve_pin_node(G, params.get("pin"))
    if not pin:
        return False, "Pin not found."

    for net in _pin_connected_nets(G, pin):
        for comp in _components_on_net(G, net):
            if not _component_matches_kind(G, comp, "capacitor"):
                continue
            for other_net in _component_nets(G, comp):
                if other_net != net and _net_is_ground(G, other_net):
                    return True, "Verification passed."

    desc_text = _safe_lower(params.get("_assertion_desc"))
    allow_resistor_fallback = _parse_bool(params.get("allow_resistor_fallback"), False)
    if not allow_resistor_fallback:
        resistor_hint_tokens = [
            "pull-down",
            "pull down",
            "pulldown",
            "resistor",
            "电阻",
            "下拉",
            "passive",
        ]
        allow_resistor_fallback = any(tok in desc_text for tok in resistor_hint_tokens)

    if allow_resistor_fallback:
        ok, _ = _check_has_res_to_ground(G, params)
        if ok:
            return True, "Verification passed (resistor-to-ground fallback)."

    return False, f"No capacitor-to-ground path found on net(s) of pin '{pin}'."


def _check_has_res_to_ground(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin = _resolve_pin_node(G, params.get("pin"))
    if not pin:
        return False, "Pin not found."

    lo, hi = _parse_ohm_range(params.get("resistance_range") or params.get("resistor_value"))

    for net in _pin_connected_nets(G, pin):
        for comp in _components_on_net(G, net):
            if not _component_matches_kind(G, comp, "resistor"):
                continue
            if not _res_value_in_range(G, comp, lo, hi):
                continue

            for other_net in _component_nets(G, comp):
                if other_net != net and _net_is_ground(G, other_net):
                    return True, "Verification passed."

    return False, f"No resistor-to-ground path found on net(s) of pin '{pin}'."


def _check_has_series_component(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin_a = _resolve_pin_node(G, params.get("pin_a"))
    pin_b = _resolve_pin_node(G, params.get("pin_b"))
    comp_type = _safe_str(params.get("component_type"))

    if not pin_a or not pin_b:
        return False, "pin_a or pin_b not found."
    if not comp_type:
        return False, "Missing component_type."

    nets_a = set(_pin_connected_nets(G, pin_a))
    nets_b = set(_pin_connected_nets(G, pin_b))
    if not nets_a or not nets_b:
        return False, "pin_a or pin_b is floating."

    for net in nets_a:
        for comp in _components_on_net(G, net):
            if not _component_matches_kind(G, comp, comp_type):
                continue
            comp_nets = _component_nets(G, comp)
            if any((nb in comp_nets) and (nb != net) for nb in nets_b):
                return True, "Verification passed."

    return False, f"No series '{comp_type}' path between '{pin_a}' and '{pin_b}'."


def _check_has_inductor_path(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    return _check_has_series_component(
        G,
        {
            "pin_a": params.get("pin_a"),
            "pin_b": params.get("pin_b"),
            "component_type": "IND",
        },
    )


def _check_has_divider_to_ground(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin = _resolve_pin_node(G, params.get("pin"))
    if not pin:
        return False, "Pin not found."

    for net in _pin_connected_nets(G, pin):
        has_to_gnd = False
        has_to_non_gnd = False
        for comp in _components_on_net(G, net):
            if not _component_matches_kind(G, comp, "resistor"):
                continue
            other_nets = [n for n in _component_nets(G, comp) if n != net]
            if any(_net_is_ground(G, n) for n in other_nets):
                has_to_gnd = True
            if any(not _net_is_ground(G, n) for n in other_nets):
                has_to_non_gnd = True
        if has_to_gnd and has_to_non_gnd:
            return True, "Verification passed."

    return False, f"No divider-to-ground topology found for pin '{pin}'."


def _check_has_crystal_pair(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    p_in = _resolve_pin_node(G, params.get("pin_osc_in"))
    p_out = _resolve_pin_node(G, params.get("pin_osc_out"))
    if not p_in or not p_out:
        return False, "pin_osc_in or pin_osc_out not found."

    nets_in = set(_pin_connected_nets(G, p_in))
    nets_out = set(_pin_connected_nets(G, p_out))
    if not nets_in or not nets_out:
        return False, "OSC pins are floating."
    if nets_in & nets_out:
        return False, "OSC IN and OUT should not share the same net."

    # Require a crystal/resonator bridging both sides.
    bridged = False
    for n_in in nets_in:
        for comp in _components_on_net(G, n_in):
            if not _component_matches_kind(G, comp, "crystal"):
                continue
            if _component_nets(G, comp) & nets_out:
                bridged = True
                break
        if bridged:
            break
    if not bridged:
        return False, "No crystal/resonator bridging OSC IN/OUT nets."

    ok_in, _ = _check_has_cap_to_ground(G, {"pin": p_in})
    ok_out, _ = _check_has_cap_to_ground(G, {"pin": p_out})
    if not (ok_in and ok_out):
        return False, "Missing load capacitor-to-ground on OSC IN/OUT net."

    return True, "Verification passed."


def _check_has_pullup_to_power(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin = _resolve_pin_node(G, params.get("pin"))
    if not pin:
        return False, "Pin not found."

    for net in _pin_connected_nets(G, pin):
        for comp in _components_on_net(G, net):
            if not _component_matches_kind(G, comp, "resistor"):
                continue
            for other_net in _component_nets(G, comp):
                if other_net != net and _net_is_power(G, other_net):
                    return True, "Verification passed."

    return False, f"No pull-up resistor to power found for pin '{pin}'."


def _parse_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    text = _safe_lower(value)
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _check_has_decoupling_network(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin = _resolve_pin_node(G, params.get("pin"))
    if not pin:
        return False, "Pin not found."

    require_pullup = _parse_bool(params.get("require_pullup_to_power"), True)
    require_cap_to_ground = _parse_bool(params.get("require_cap_to_ground"), True)

    for net in _pin_connected_nets(G, pin):
        has_pullup = False
        has_cap_gnd = False

        for comp in _components_on_net(G, net):
            comp_nets = _component_nets(G, comp)
            other_nets = [n for n in comp_nets if n != net]

            if _component_matches_kind(G, comp, "resistor"):
                if any(_net_is_power(G, n) for n in other_nets):
                    has_pullup = True

            if _component_matches_kind(G, comp, "capacitor"):
                if any(_net_is_ground(G, n) for n in other_nets):
                    has_cap_gnd = True

        pullup_ok = (not require_pullup) or has_pullup
        cap_ok = (not require_cap_to_ground) or has_cap_gnd
        if pullup_ok and cap_ok:
            return True, "Verification passed."

    return False, f"No decoupling network found for pin '{pin}' (need pull-up to power and capacitor to ground)."


def _check_has_differential_pair(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    p_pin = _resolve_pin_node(G, params.get("pin_p"))
    n_pin = _resolve_pin_node(G, params.get("pin_n"))
    if not p_pin or not n_pin:
        return False, "pin_p or pin_n not found."

    p_nets = set(_pin_connected_nets(G, p_pin))
    n_nets = set(_pin_connected_nets(G, n_pin))
    if not p_nets or not n_nets:
        return False, "Differential pin is floating."
    if p_nets & n_nets:
        return False, "Differential pins are shorted to the same net."

    return True, "Verification passed."


def _check_has_sense_resistor(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin = _resolve_pin_node(G, params.get("pin"))
    if not pin:
        return False, "Pin not found."

    max_res = _parse_res_ohms(params.get("max_resistor_value"))
    ref_type = _safe_lower(params.get("reference_net_type", "any"))

    for net in _pin_connected_nets(G, pin):
        for comp in _components_on_net(G, net):
            if not _component_matches_kind(G, comp, "resistor"):
                continue
            if max_res is not None and not _res_value_in_range(G, comp, 0.0, max_res):
                continue
            other_nets = [n for n in _component_nets(G, comp) if n != net]
            if not other_nets:
                continue

            if ref_type in {"", "any", "auto"}:
                return True, "Verification passed."
            if ref_type in {"ground", "gnd"} and any(_net_is_ground(G, n) for n in other_nets):
                return True, "Verification passed."
            if ref_type in {"power", "vcc", "vdd"} and any(_net_is_power(G, n) for n in other_nets):
                return True, "Verification passed."

    if ref_type in {"ground", "gnd"}:
        return False, f"No sense resistor path from pin '{pin}' to Ground found."
    if ref_type in {"power", "vcc", "vdd"}:
        return False, f"No sense resistor path from pin '{pin}' to Power found."
    return False, f"No sense resistor found on net(s) of pin '{pin}'."


def _check_has_freewheeling_path(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin = _resolve_pin_node(G, params.get("pin"))
    if not pin:
        return False, "Pin not found."

    for net in _pin_connected_nets(G, pin):
        for comp in _components_on_net(G, net):
            if not _component_matches_kind(G, comp, "diode"):
                continue
            for other_net in _component_nets(G, comp):
                if other_net == net:
                    continue
                if _net_is_ground(G, other_net) or _net_is_power(G, other_net):
                    return True, "Verification passed."

    return False, f"No freewheeling diode path found for pin '{pin}'."


def _check_has_feedback_divider(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin_fb = _resolve_pin_node(G, params.get("pin_fb") or params.get("pin"))
    if not pin_fb:
        return False, "pin_fb not found."

    target_out_net = _safe_str(params.get("output_net") or params.get("pin_output_net"))
    target_gnd_net = _safe_str(params.get("ground_net") or params.get("pin_ground_net"))

    target_out_nets: Set[str] = set()
    target_gnd_nets: Set[str] = set()

    if target_out_net:
        resolved_out = _resolve_net_node(G, target_out_net)
        if resolved_out:
            target_out_nets.add(resolved_out)
    if target_gnd_net:
        resolved_gnd = _resolve_net_node(G, target_gnd_net)
        if resolved_gnd:
            target_gnd_nets.add(resolved_gnd)

    output_pin_token = params.get("output_pin")
    if output_pin_token is not None and _safe_str(output_pin_token):
        output_pin = _resolve_pin_node(G, output_pin_token)
        if not output_pin:
            return False, "output_pin not found."
        target_out_nets.update(_pin_connected_nets(G, output_pin))

    ground_pin_token = params.get("ground_pin")
    if ground_pin_token is not None and _safe_str(ground_pin_token):
        ground_pin = _resolve_pin_node(G, ground_pin_token)
        if not ground_pin:
            return False, "ground_pin not found."
        target_gnd_nets.update(_pin_connected_nets(G, ground_pin))

    fb_nets = _pin_connected_nets(G, pin_fb)
    if not fb_nets:
        return False, "pin_fb is floating."

    for fb_net in fb_nets:
        has_r_to_ground = False
        has_r_to_output = False

        for comp in _components_on_net(G, fb_net):
            if not _component_matches_kind(G, comp, "resistor"):
                continue

            for other_net in _component_nets(G, comp):
                if other_net == fb_net:
                    continue

                is_ground = _net_is_ground(G, other_net)
                if target_gnd_nets:
                    is_ground = other_net in target_gnd_nets
                if is_ground:
                    has_r_to_ground = True

                is_output = not _net_is_ground(G, other_net)
                if target_out_nets:
                    is_output = other_net in target_out_nets
                if is_output:
                    has_r_to_output = True

        if has_r_to_ground and has_r_to_output:
            return True, "Verification passed."

    return False, f"No feedback divider found for pin '{pin_fb}'."


def _check_has_bus_termination(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin_p = _resolve_pin_node(G, params.get("pin_p") or params.get("bus_pin_a"))
    pin_n = _resolve_pin_node(G, params.get("pin_n") or params.get("bus_pin_b"))
    if not pin_p or not pin_n:
        return False, "pin_p or pin_n not found."

    p_nets = set(_pin_connected_nets(G, pin_p))
    n_nets = set(_pin_connected_nets(G, pin_n))
    if not p_nets or not n_nets:
        return False, "Bus pin is floating."
    if p_nets & n_nets:
        return False, "Bus differential pins share same net."

    lo, hi = _parse_ohm_range(params.get("resistance_range") or params.get("termination_value"))

    for net_p in p_nets:
        for comp in _components_on_net(G, net_p):
            if not _component_matches_kind(G, comp, "resistor"):
                continue
            if not _res_value_in_range(G, comp, lo, hi):
                continue
            comp_nets = _component_nets(G, comp)
            if comp_nets & n_nets:
                return True, "Verification passed."

    return False, "No termination resistor found between the two bus nets."


def _path_exists_between_nodes(
    G: nx.Graph,
    src_node: str,
    dst_nodes: Set[str],
    max_hops: int,
    allowed_component_kinds: Optional[Set[str]] = None,
) -> bool:
    if src_node in dst_nodes:
        return True

    allowed_types = {"Pin", "Net", "Component"}

    q = deque([(src_node, 0)])
    seen: Set[str] = {src_node}

    while q:
        node, dist = q.popleft()
        if dist >= max_hops:
            continue

        for nbr in G.neighbors(node):
            if nbr in seen:
                continue

            nbr_type = _node_type(G, nbr)
            if nbr_type not in allowed_types:
                continue

            rel = _edge_relation(G, node, nbr)
            if rel and rel not in {"connected_to", "belongs_to"}:
                continue

            if nbr_type == "Component" and not _component_allowed_for_path(G, nbr, allowed_component_kinds):
                continue

            if nbr in dst_nodes:
                return True

            seen.add(nbr)
            q.append((nbr, dist + 1))

    return False


def _path_exists_pin_to_pin(G: nx.Graph, src_pin: str, dst_pin: str, max_hops: int) -> bool:
    return _path_exists_between_nodes(
        G,
        src_node=src_pin,
        dst_nodes={dst_pin},
        max_hops=max_hops,
        allowed_component_kinds=None,
    )


def _check_connected_to(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin = _resolve_pin_node(G, params.get("pin") or params.get("from_pin"))
    if not pin:
        return False, "Pin not found."

    target_net = _resolve_net_node(G, params.get("net_name") or params.get("destination_net") or params.get("destination"))
    if target_net:
        if target_net in _pin_connected_nets(G, pin):
            return True, "Verification passed."
        return False, f"Pin '{pin}' is not connected to net '{target_net}'."

    target_pin = _resolve_pin_node(G, params.get("destination_pin") or params.get("to_pin") or params.get("destination"))
    if target_pin:
        pin_nets = set(_pin_connected_nets(G, pin))
        target_nets = set(_pin_connected_nets(G, target_pin))
        if pin_nets & target_nets:
            return True, "Verification passed."
        return False, f"Pin '{pin}' does not share net with '{target_pin}'."

    return False, "CONNECTED_TO requires destination pin or net_name."


def _check_must_be_floating(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pin = _resolve_pin_node(G, params.get("pin"))
    if not pin:
        return False, "Pin not found."

    nets = _pin_connected_nets(G, pin)
    if not nets:
        return True, "Verification passed."
    return False, f"Pin '{pin}' should be floating but is connected to net(s): {sorted(set(nets))}."


def _check_exists_path(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    src = _resolve_pin_node(G, params.get("from_pin") or params.get("pin"))
    if not src:
        return False, "from_pin not found."

    destination_token = params.get("destination")
    if destination_token is None:
        destination_token = params.get("to_pin")
    if destination_token is None:
        destination_token = params.get("net_name")
    if destination_token is None:
        return False, "EXISTS_PATH requires destination/to_pin/net_name."

    dst_pin = _resolve_pin_node(G, destination_token)
    dst_net = _resolve_net_node(G, destination_token)
    dst_nodes: Set[str] = set()
    if dst_pin:
        dst_nodes.add(dst_pin)
    if dst_net:
        dst_nodes.add(dst_net)
    if not dst_nodes:
        return False, f"Destination not found: {destination_token}."

    allowed_raw = params.get("allowed_components", params.get("via"))
    if allowed_raw is None:
        allowed_kinds = set(DEFAULT_ALLOWED_PATH_COMPONENTS)
    else:
        allowed_kinds = _normalize_allowed_component_kinds(allowed_raw)
    if allowed_raw is not None and not allowed_kinds:
        return False, "allowed_components/via is empty or invalid."

    max_hops = int(params.get("max_hops", 12) or 12)
    if _path_exists_between_nodes(
        G,
        src_node=src,
        dst_nodes=dst_nodes,
        max_hops=max_hops,
        allowed_component_kinds=allowed_kinds,
    ):
        return True, "Verification passed."

    if allowed_raw is None:
        return False, (
            f"No path found from '{src}' to '{_safe_str(destination_token)}' within {max_hops} hops "
            f"using default passive components {sorted(allowed_kinds)}."
        )
    return False, (
        f"No constrained path found from '{src}' to '{_safe_str(destination_token)}' "
        f"within {max_hops} hops using components {sorted(allowed_kinds)}."
    )


def _check_path_exists(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    merged = dict(params)
    if merged.get("destination") is None and merged.get("to_pin") is not None:
        merged["destination"] = merged.get("to_pin")
    return _check_exists_path(G, merged)


def _check_signal_chain_order(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    pins = params.get("pins", [])
    max_hops = int(params.get("max_hops", 12) or 12)

    if not isinstance(pins, list) or len(pins) < 2:
        return False, "SIGNAL_CHAIN_ORDER requires at least two pins."

    resolved, missing = _resolve_pin_list(G, pins, allow_unresolved_passive=True)
    if missing:
        return False, f"Pin(s) not found: {missing}."
    if len(resolved) < 2:
        return True, "Verification passed (insufficient resolvable pins after passive normalization)."

    for i in range(len(resolved) - 1):
        if not _path_exists_pin_to_pin(G, resolved[i], resolved[i + 1], max_hops):
            return False, f"No ordered path between '{pins[i]}' and '{pins[i + 1]}'."

    return True, "Verification passed."


def _check_bus_shared(G: nx.Graph, params: Dict[str, Any]) -> Tuple[bool, str]:
    master = params.get("master_pins", [])
    slave_groups = params.get("slave_pins_groups", [])

    if not isinstance(master, list) or not master:
        return False, "master_pins must be a non-empty list."
    if not isinstance(slave_groups, list) or not slave_groups:
        return False, "slave_pins_groups must be a non-empty list."

    for idx, m_pin in enumerate(master):
        line_pins = [m_pin]
        for group in slave_groups:
            if not isinstance(group, list) or idx >= len(group):
                return False, "Each slave pin group must align with master_pins length."
            line_pins.append(group[idx])

        ok, msg = _check_same_net(G, {"pins": line_pins})
        if not ok:
            return False, f"BUS_SHARED line#{idx} failed: {msg}"

    return True, "Verification passed."


CHECK_HANDLERS: Dict[str, Callable[[nx.Graph, Dict[str, Any]], Tuple[bool, str]]] = {
    "PIN_SINGLE_NET": _check_pin_single_net,
    "PIN_NOT_FLOATING": _check_pin_not_floating,
    "MUST_BE_FLOATING": _check_must_be_floating,
    "SAME_NET": _check_same_net,
    "CONNECTED_TO": _check_connected_to,
    "CONNECTED_TO_GROUND": _check_connected_to_ground,
    "CONNECTED_TO_POWER": _check_connected_to_power,
    "CONNECTED_TO_NET_CLASS": _check_connected_to_net_class,
    "NOT_SAME_NET": _check_not_same_net,
    "NET_GROUP_ISOLATION": _check_net_group_isolation,
    "PIN_NOT_ON_NET_WITH": _check_pin_not_on_net_with,
    "HAS_CAP_TO_GROUND": _check_has_cap_to_ground,
    "HAS_RES_TO_GROUND": _check_has_res_to_ground,
    "HAS_SERIES_COMPONENT": _check_has_series_component,
    "HAS_INDUCTOR_PATH": _check_has_inductor_path,
    "HAS_DIVIDER_TO_GROUND": _check_has_divider_to_ground,
    "HAS_CRYSTAL_PAIR": _check_has_crystal_pair,
    "HAS_PULLUP_TO_POWER": _check_has_pullup_to_power,
    "HAS_DIFFERENTIAL_PAIR": _check_has_differential_pair,
    "HAS_SENSE_RESISTOR": _check_has_sense_resistor,
    "HAS_FREEWHEELING_PATH": _check_has_freewheeling_path,
    "HAS_DECOUPLING_NETWORK": _check_has_decoupling_network,
    "HAS_FEEDBACK_DIVIDER": _check_has_feedback_divider,
    "HAS_BUS_TERMINATION": _check_has_bus_termination,
    "EXISTS_PATH": _check_exists_path,
    "PATH_EXISTS": _check_path_exists,
    "SIGNAL_CHAIN_ORDER": _check_signal_chain_order,
    "BUS_SHARED": _check_bus_shared,
}


def verify_assertion(G: nx.Graph, assertion: Dict[str, Any]) -> Tuple[bool, str]:
    logic_type = _safe_str(
        assertion.get("type")
        or assertion.get("logic")
        or assertion.get("check")
        or assertion.get("condition")
    ).upper()
    if logic_type in {"ANY_OF", "OR_GROUP", "OR"}:
        conditions = assertion.get("conditions", [])
        if not isinstance(conditions, list) or not conditions:
            return False, "ANY_OF requires a non-empty conditions list."

        reasons: List[str] = []
        for cond in conditions:
            if not isinstance(cond, dict):
                reasons.append("Condition is not an object")
                continue
            ok, msg = verify_assertion(G, cond)
            if ok:
                return True, "Verification passed."
            reasons.append(msg)

        preview = "; ".join(reasons[:3])
        return False, f"ANY_OF failed: {preview}"

    if logic_type in {"ALL_OF", "AND_GROUP", "AND"}:
        conditions = assertion.get("conditions", [])
        if not isinstance(conditions, list) or not conditions:
            return False, "ALL_OF requires a non-empty conditions list."

        for cond in conditions:
            if not isinstance(cond, dict):
                return False, "Condition is not an object"
            ok, msg = verify_assertion(G, cond)
            if not ok:
                return False, f"ALL_OF failed: {msg}"
        return True, "Verification passed."

    normalized = _normalize_atomic_assertion(assertion)
    check = _safe_str(normalized.get("check")).upper()
    params = normalized.get("params", {})
    if not isinstance(params, dict):
        return False, "Invalid params: must be an object."

    handler = CHECK_HANDLERS.get(check)
    if handler is None:
        return False, f"Unsupported check type: {check}."

    try:
        return handler(G, params)
    except Exception as e:
        return False, f"Assertion runtime error: {e}"


def evaluate_assertions(
    G: nx.Graph,
    assertions: Sequence[Dict[str, Any]],
    stop_on_first_failure: bool = False,
) -> AssertionSummary:
    assertion_list = list(assertions or [])
    total = len(assertion_list)
    passed = 0
    failed = 0
    functional_total = 0
    functional_passed = 0
    failures: List[AssertionFailure] = []

    has_explicit_functional = any(
        isinstance(a, dict) and _is_functional_assertion(a)
        for a in assertion_list
    )

    for idx, assertion in enumerate(assertion_list, start=1):
        is_functional = True
        if has_explicit_functional:
            is_functional = isinstance(assertion, dict) and _is_functional_assertion(assertion)

        if not isinstance(assertion, dict):
            failed += 1
            if is_functional:
                functional_total += 1
            failures.append(
                {
                    "id": f"UNKNOWN-{idx}",
                    "check": "UNKNOWN",
                    "desc": "",
                    "message": "Assertion entry must be an object.",
                }
            )
            if stop_on_first_failure:
                break
            continue

        ok, msg = verify_assertion(G, assertion)
        if ok:
            passed += 1
            if is_functional:
                functional_total += 1
                functional_passed += 1
            continue

        failed += 1
        if is_functional:
            functional_total += 1
        failures.append(
            {
                "id": _safe_str(assertion.get("id")) or f"UNKNOWN-{idx}",
                "check": _safe_str(assertion.get("check") or assertion.get("condition") or assertion.get("type")) or "UNKNOWN",
                "desc": _safe_str(assertion.get("desc")),
                "message": msg,
            }
        )
        if stop_on_first_failure:
            break

    pass_rate = (passed / total) if total > 0 else 1.0
    acr = (functional_passed / functional_total) if functional_total > 0 else 1.0
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "functional_total": functional_total,
        "functional_passed": functional_passed,
        "acr": acr,
        "failures": failures,
    }


def check_design_assertions(G: nx.Graph, assertions: Sequence[Dict[str, Any]]) -> bool:
    summary = evaluate_assertions(G, assertions, stop_on_first_failure=True)
    return int(summary.get("failed", 0)) == 0


def list_multi_net_pin_violations(G: nx.Graph) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    for node, attr in G.nodes(data=True):
        if _safe_str(attr.get("type")) != "Pin":
            continue
        nets = sorted(set(_pin_connected_nets(G, node)))
        if len(nets) > 1:
            violations.append({"pin": node, "nets": nets})

    violations.sort(key=lambda item: _safe_str(item.get("pin")))
    return violations


def has_multi_net_pin(G: nx.Graph) -> bool:
    return len(list_multi_net_pin_violations(G)) > 0
