import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
from networkx.readwrite import json_graph


def extract_connection_set_raw(G: nx.Graph) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract the Connection Set from the final graph in Refdes.PinName format.
    """
    components: List[Dict[str, Any]] = []
    nets: List[Dict[str, Any]] = []

    def _parent_component(pin_id: str) -> Optional[str]:
        if not G.has_node(pin_id):
            return None
        for nbr in G.neighbors(pin_id):
            if (
                G.nodes[nbr].get("type") == "Component"
                and G.get_edge_data(pin_id, nbr).get("relation") == "Belongs_To"
            ):
                return str(nbr)
        return None

    def _pin_ref(pin_id: str) -> str:
        comp = _parent_component(pin_id)
        if not comp:
            return pin_id
        pin_attr = G.nodes[pin_id]
        pin_name = str(pin_attr.get("name", "")).strip() or str(pin_attr.get("pin_number", "")).strip() or "PIN"
        return f"{comp}.{pin_name}"

    for node, attr in G.nodes(data=True):
        if attr.get("type") == "Component":
            components.append(
                {
                    "refdes": node,
                    "part_number": attr.get("part_number", ""),
                }
            )

        if attr.get("type") == "Net":
            connected_pin_nodes = [
                nbr
                for nbr in G.neighbors(node)
                if G.nodes[nbr].get("type") == "Pin"
                and G.get_edge_data(node, nbr).get("relation") == "Connected_To"
            ]
            connected_pins = sorted({_pin_ref(pin) for pin in connected_pin_nodes})

            if connected_pins:
                nets.append(
                    {
                        "name": attr.get("name", node),
                        "pins": connected_pins,
                    }
                )

    components.sort(key=lambda x: x["refdes"])
    nets.sort(key=lambda x: x["name"])

    return {"components": components, "nets": nets}


def serialize_graph(G: nx.Graph) -> Dict[str, Any]:
    """
    将 NetworkX 图序列化为 node-link JSON 结构。
    """
    return json_graph.node_link_data(G, edges="links")


def deserialize_graph(data: Dict[str, Any]) -> nx.Graph:
    """
    从 node-link JSON 恢复 NetworkX 图。
    """
    return json_graph.node_link_graph(data, edges="links")


def _safe_str(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _normalize_pins_list(value: object) -> List[str]:
    if isinstance(value, str):
        token = value.strip()
        return [token] if token else []

    if not isinstance(value, Iterable):
        return []

    out: List[str] = []
    for item in value:
        token = _safe_str(item)
        if token:
            out.append(token)
    return out


def _infer_net_type(net_name: str) -> str:
    upper = net_name.upper()
    if any(k in upper for k in ["GND", "AGND", "DGND", "PGND", "GROUND", "VSS"]):
        return "Ground"
    if any(k in upper for k in ["VCC", "VDD", "VIN", "VOUT", "VBAT", "VDDA", "VDDIO", "VREF"]):
        return "Power"
    return "Signal"


def _infer_pin_electrical_type(pin_name: str) -> str:
    upper = pin_name.upper()
    if any(k in upper for k in ["GND", "AGND", "DGND", "PGND", "GROUND", "VSS"]):
        return "Ground"
    if any(k in upper for k in ["VCC", "VDD", "VIN", "VOUT", "VBAT", "VDDA", "VDDIO", "VREF"]):
        return "PowerIn"
    return "Passive"


def _parse_pin_token(raw_pin_id: str) -> Tuple[Optional[str], str, str]:
    """
    Parse pin token and return (component_refdes, pin_name, pin_number).
    """
    token = _safe_str(raw_pin_id)
    if not token:
        return None, "PIN", ""

    if "." in token:
        comp, ident = token.split(".", 1)
        ident = ident.strip() or "PIN"
        pin_num = ident if ident.isdigit() else ""
        return comp.strip() or None, ident, pin_num

    m = re.match(r"^(?P<comp>.+)_Pin(?P<pnum>[^_]+)$", token)
    if m:
        comp = m.group("comp").strip() or None
        pnum = m.group("pnum").strip()
        pname = pnum if pnum else "PIN"
        return comp, pname, pnum

    return None, "PIN", ""


def build_graph_from_connection_set(
    connection_set: Dict[str, Any],
    module_id: str = "baseline_module",
) -> nx.Graph:
    """
    Build a minimal-but-valid design graph from connection_set for evaluation fallback.

    This graph is not a full reconstruction of the design process. It is intended to:
    - avoid evaluator crashes when graph_snapshot is absent,
    - enable pin normalization and structure checks,
    - provide best-effort ERC evaluation inputs.
    """
    G = nx.Graph()

    G.add_node(
        module_id,
        type="Module",
        name=module_id,
        description="Auto-built from connection_set",
    )

    raw_components = connection_set.get("components", []) if isinstance(connection_set, dict) else []
    raw_nets = connection_set.get("nets", []) if isinstance(connection_set, dict) else []

    components: Dict[str, Dict[str, str]] = {}
    if isinstance(raw_components, list):
        for item in raw_components:
            if not isinstance(item, dict):
                continue
            refdes = _safe_str(item.get("refdes"))
            if not refdes:
                continue
            components[refdes] = {
                "part_number": _safe_str(item.get("part_number")),
            }

    # Build nets and infer missing components from pin tokens.
    if isinstance(raw_nets, list):
        for idx, net in enumerate(raw_nets, start=1):
            if not isinstance(net, dict):
                continue

            net_name = _safe_str(net.get("name")) or f"NET_{idx}"
            net_id = net_name
            if not G.has_node(net_id):
                net_type = _infer_net_type(net_name)
                G.add_node(
                    net_id,
                    type="Net",
                    name=net_name,
                    scope="Local",
                    net_type=net_type,
                    net_class=net_type,
                    semantic_role="Auto-built from connection_set",
                    source_pin="",
                )
                G.add_edge(net_id, module_id, relation="Belongs_To")

            pins = _normalize_pins_list(net.get("pins", []))
            for raw_pin in pins:
                comp_id, pin_name, pin_number = _parse_pin_token(raw_pin)

                if comp_id and comp_id not in components:
                    components[comp_id] = {"part_number": ""}

                pin_id = raw_pin
                if not G.has_node(pin_id):
                    G.add_node(
                        pin_id,
                        type="Pin",
                        name=pin_name,
                        pin_number=pin_number,
                        electrical_type=_infer_pin_electrical_type(pin_name),
                        description="Auto-built from connection_set",
                    )

                if comp_id:
                    if not G.has_node(comp_id):
                        G.add_node(
                            comp_id,
                            type="Component",
                            refdes=comp_id,
                            part_number=components.get(comp_id, {}).get("part_number", ""),
                            description="Auto-built from connection_set",
                        )
                        G.add_edge(comp_id, module_id, relation="Belongs_To")

                    if not G.has_edge(pin_id, comp_id):
                        G.add_edge(pin_id, comp_id, relation="Belongs_To")

                if not G.has_edge(pin_id, net_id):
                    G.add_edge(pin_id, net_id, relation="Connected_To")

    # Ensure all declared components exist.
    for refdes, meta in components.items():
        if not G.has_node(refdes):
            G.add_node(
                refdes,
                type="Component",
                refdes=refdes,
                part_number=meta.get("part_number", ""),
                description="Auto-built from connection_set",
            )
            G.add_edge(refdes, module_id, relation="Belongs_To")
        elif G.nodes[refdes].get("type") == "Component":
            if meta.get("part_number") and not G.nodes[refdes].get("part_number"):
                G.nodes[refdes]["part_number"] = meta["part_number"]

    return G
