from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


def _pin_number_as_int(value: object) -> int:
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return 0


def _parent_component(G: nx.Graph, pin_id: str) -> Optional[str]:
    if not G.has_node(pin_id):
        return None

    for nbr in G.neighbors(pin_id):
        if (
            G.nodes[nbr].get("type") == "Component"
            and G.get_edge_data(pin_id, nbr).get("relation") == "Belongs_To"
        ):
            return nbr
    return None


def _pin_identifier(pin_attr: Dict[str, object]) -> str:
    name = str(pin_attr.get("name", "")).strip()
    if name:
        return name.replace(" ", "_")

    number = str(pin_attr.get("pin_number", "")).strip()
    if number:
        return number

    return "PIN"


def _strip_suffix_index(token: str) -> str:
    return re.sub(r"_\d+$", "", token)


def normalize_pin_id_from_graph(G: nx.Graph, raw_pin_id: str) -> str:
    """
    Convert graph pin node id (e.g. U1_Pin3) to GT style id (e.g. U1.VIN).
    """
    pin_map = resolve_duplicate_pin_names(G)
    return pin_map.get(raw_pin_id, raw_pin_id)


def resolve_duplicate_pin_names(G: nx.Graph) -> Dict[str, str]:
    """
    Build full mapping raw_pin_id -> normalized_pin_id.

    If one component has duplicate pin names, append _N sorted by pin_number.
    """
    mapping: Dict[str, str] = {}

    comp_nodes = [
        node
        for node, attr in G.nodes(data=True)
        if attr.get("type") == "Component"
    ]

    for comp_id in comp_nodes:
        pin_nodes = [
            nbr
            for nbr in G.neighbors(comp_id)
            if (
                G.nodes[nbr].get("type") == "Pin"
                and G.get_edge_data(nbr, comp_id).get("relation") == "Belongs_To"
            )
        ]

        grouped: Dict[str, List[Tuple[str, int]]] = {}
        for pin_id in pin_nodes:
            pin_attr = G.nodes[pin_id]
            ident = _pin_identifier(pin_attr)
            grouped.setdefault(ident, []).append(
                (pin_id, _pin_number_as_int(pin_attr.get("pin_number", "")))
            )

        for ident, entries in grouped.items():
            entries.sort(key=lambda item: item[1])

            if len(entries) == 1:
                pin_id = entries[0][0]
                mapping[pin_id] = f"{comp_id}.{ident}"
                continue

            # If ident already has explicit index and all are distinct, keep them.
            normalized_base = _strip_suffix_index(ident)
            has_explicit_index = bool(re.search(r"_\d+$", ident))
            if has_explicit_index:
                for pin_id, _ in entries:
                    mapping[pin_id] = f"{comp_id}.{ident}"
                continue

            for idx, (pin_id, _) in enumerate(entries, start=1):
                mapping[pin_id] = f"{comp_id}.{normalized_base}_{idx}"

    return mapping


def normalize_connection_set(G: nx.Graph, raw_connection_set: Dict[str, object]) -> Dict[str, object]:
    """
    Normalize all pin ids in a raw connection set.
    """
    if not isinstance(raw_connection_set, dict):
        return {"components": [], "nets": []}

    pin_map = resolve_duplicate_pin_names(G)
    pin_map_upper = {k.upper(): v for k, v in pin_map.items()}

    raw_components = raw_connection_set.get("components", [])
    components = raw_components if isinstance(raw_components, list) else []

    raw_nets = raw_connection_set.get("nets", [])
    if not isinstance(raw_nets, list):
        raw_nets = []

    normalized_nets: List[Dict[str, object]] = []
    for net in raw_nets:
        if not isinstance(net, dict):
            continue

        raw_pins_obj: Any = net.get("pins", [])
        if isinstance(raw_pins_obj, str):
            raw_pins = [raw_pins_obj]
        elif isinstance(raw_pins_obj, list):
            raw_pins = [str(p) for p in raw_pins_obj]
        else:
            raw_pins = []

        normalized_pins = []
        for raw_pin in raw_pins:
            key = str(raw_pin)
            mapped = pin_map.get(key)
            if mapped is None:
                mapped = pin_map_upper.get(key.upper())
            normalized_pins.append(mapped if mapped is not None else key)

        normalized_pins = sorted(p for p in normalized_pins if str(p).strip())
        normalized_nets.append(
            {
                "name": str(net.get("name", "")),
                "pins": normalized_pins,
            }
        )

    return {
        "components": components,
        "nets": normalized_nets,
    }
