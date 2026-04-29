import re
from typing import Any, Dict, List, Optional

import yaml

from tools.build_graph import get_component_data


def _build_focus_layer_extras(part_number: str, include_ref_circuits: bool = True) -> Dict[str, Any]:
    """
    Inject Focus Layer extras for the expanded component: reference_circuits and erc_rules_summary.
    """
    extras: Dict[str, Any] = {}
    if not part_number:
        return extras

    lib_data = get_component_data(part_number)
    if not lib_data:
        return extras

    if include_ref_circuits:
        ref_circuits = lib_data.get("reference_circuits", [])
        if ref_circuits:
            extras["reference_circuits"] = [
                {
                    "name": rc.get("name", "default"),
                    "topology_nets": rc.get("topology_nets", []),
                    "required_passives": rc.get("required_passives", []),
                }
                for rc in ref_circuits
            ]

    erc_rules = lib_data.get("erc_rules", [])
    if erc_rules:
        extras["erc_rules_summary"] = [
            {
                "rule_type": rule.get("rule_type"),
                "description": rule.get("description", ""),
            }
            for rule in erc_rules
        ]

    return extras


def _pin_summary(G, pin_id: str) -> str:
    """
    Generate a pin summary string: Comp.PinNum(PinName).
    """
    pin_attr = G.nodes[pin_id]
    comp = None
    for nbr in G.neighbors(pin_id):
        if G.nodes[nbr].get("type") == "Component" and G.get_edge_data(pin_id, nbr).get("relation") == "Belongs_To":
            comp = nbr
            break

    pin_num = pin_attr.get("pin_number", "?")
    pin_name = pin_attr.get("name", "")
    if comp:
        return f"{comp}.Pin{pin_num}({pin_name})"
    return f"Pin{pin_num}({pin_name})"


def _is_passive_component(G, comp_id: str) -> bool:
    """
    Determine whether a component is passive, used to decide how compactly its pins should be displayed.
    """
    if not G.has_node(comp_id):
        return False

    attr = G.nodes[comp_id]
    part_type = str(attr.get("part_type", "")).strip().lower()
    if part_type in {
        "resistor",
        "capacitor",
        "inductor",
        "diode",
        "led",
        "passive",
        "ferritebead",
        "resonator",
        "crystal",
        "fuse",
    }:
        return True

    part_number = str(attr.get("part_number", "")).strip().upper()
    if part_number.startswith(("RES_", "CAP_", "IND_", "DIODE_", "LED_", "CRYSTAL_", "FUSE_")):
        return True

    return bool(re.match(r"^(R|C|L|D|Y|F)\d+$", str(comp_id), flags=re.IGNORECASE))


def _is_pin_critical_for_visibility(pin_attr: Dict[str, Any]) -> bool:
    """
    Determine whether a pin is critical and should remain visible by default.
    """
    electrical = str(pin_attr.get("electrical_type", "")).strip().lower()
    if any(k in electrical for k in ["power", "ground", "supply"]):
        return True

    name = str(pin_attr.get("name", "")).strip().lower()
    keep_tokens = [
        "vdd", "vcc", "vss", "gnd", "agnd", "dgnd", "vbat", "vref", "vin", "vout",
        "rst", "reset", "nrst", "boot", "swd", "swclk", "swdio", "jtag", "osc", "xtal",
        "scl", "sda", "tx", "rx",
    ]
    return any(tok in name for tok in keep_tokens)


def _pin_ref_sort_key(pin_ref: Dict[str, str]) -> tuple:
    pin_num = str(pin_ref.get("pin_number", "")).strip()
    pin_name = str(pin_ref.get("pin_name", "")).strip()
    if pin_num.isdigit():
        return (0, int(pin_num), pin_name)
    return (1, pin_num, pin_name)


def inspect_design_overview(G) -> str:
    """
    Export the global design overview for the initial visible scope.

    Under the v4 Phase 0 schema, VirtualPort nodes are not inserted into the graph, so shared_nets/global_nets
    are additionally exported.
    """
    state = {
        "modules": [],
    }

    modules = [n for n, attr in G.nodes(data=True) if attr.get("type") == "Module"]
    for module_id in sorted(modules):
        module_attr = G.nodes[module_id]
        vports = []
        for nbr in G.neighbors(module_id):
            if (
                G.nodes[nbr].get("type") == "VirtualPort"
                and G.get_edge_data(module_id, nbr).get("relation") == "Belongs_To"
            ):
                vp_attr = G.nodes[nbr]
                vports.append(
                    {
                        "name": vp_attr.get("name", nbr),
                        "electrical_type": vp_attr.get("electrical_type", "Signal"),
                    }
                )

        state["modules"].append(
            {
                "module_id": module_id,
                "description": module_attr.get("description", ""),
                "virtual_ports": sorted(vports, key=lambda x: x["name"]),
            }
        )

    shared_nets = G.graph.get("shared_nets")
    if isinstance(shared_nets, dict) and shared_nets:
        state["shared_nets"] = [
            {
                "name": name,
                "type": meta.get("type", "Signal"),
                "modules": meta.get("modules", []),
            }
            for name, meta in sorted(shared_nets.items(), key=lambda kv: kv[0])
        ]

    global_nets = []
    for node, attr in G.nodes(data=True):
        if attr.get("type") != "Net":
            continue
        if attr.get("scope") != "Global":
            continue

        global_nets.append(
            {
                "name": attr.get("name", node),
                "type": attr.get("net_type", "Signal"),
            }
        )

    if global_nets:
        state["global_nets"] = sorted(global_nets, key=lambda x: x["name"])

    return yaml.dump(
        state,
        sort_keys=False,
        default_flow_style=False,
        Dumper=yaml.SafeDumper,
        allow_unicode=True,
    )


def inspect_module_state(
    G,
    module_id: str,
    expand_components: Optional[List[str]] = None,
    include_nets: bool = True,
    include_ref_circuits: bool = True,
) -> str:
    """
    Export the module state as YAML, so the LLM can read it as low-noise context.

    Output includes:
    1) virtual_ports.
    2) components, with optional pin expansion.
    3) nets, including local nets and global nets.
    """
    if expand_components is None:
        expand_components = []

    state = {
        "module": module_id,
        "virtual_ports": [],
        "components": [],
        "nets": [],
    }

    # First extract module-level entities.
    for node, attr in G.nodes(data=True):
        node_type = attr.get("type")

        # 1) Virtual ports.
        if node_type == "VirtualPort" and G.has_edge(node, module_id):
            state["virtual_ports"].append(
                {
                    "name": attr.get("name", node),
                    "electrical_type": attr.get("electrical_type", "Signal"),
                }
            )

        # 2) Components, with optional pin expansion.
        elif node_type == "Component" and G.has_edge(node, module_id):
            comp_info = {
                "refdes": node,
                "part": attr.get("part_number", attr.get("part_type", "Passive")),
            }

            is_passive = _is_passive_component(G, node)

            if node in expand_components:
                pins = []
                comp_pin_nodes = [
                    neighbor
                    for neighbor in G.neighbors(node)
                    if (
                        G.nodes[neighbor].get("type") == "Pin"
                        and G.get_edge_data(neighbor, node).get("relation") == "Belongs_To"
                    )
                ]

                hidden_pin_count = 0
                connected_pin_count = 0
                fold_candidates: List[Dict[str, str]] = []

                for neighbor in comp_pin_nodes:
                    pin_attr = G.nodes[neighbor]
                    pin_nets = [n for n in G.neighbors(neighbor) if G.nodes[n].get("type") == "Net"]
                    if pin_nets:
                        connected_pin_count += 1
                    elif not _is_pin_critical_for_visibility(pin_attr):
                        fold_candidates.append(
                            {
                                "pin_number": str(pin_attr.get("pin_number", "")),
                                "pin_name": str(pin_attr.get("name", "")),
                            }
                        )

                    if bool(pin_attr.get("hidden_in_view", False)):
                        hidden_pin_count += 1
                        continue

                    if is_passive:
                        pin_entry = {
                            "pin_name": pin_attr.get("name"),
                            "connected_nets": pin_nets if pin_nets else "UNCONNECTED",
                        }
                    else:
                        if pin_nets:
                            # For connected pins, keep only the minimal information to reduce context tokens.
                            pin_entry = {
                                "pin_number": pin_attr.get("pin_number"),
                                "pin_name": pin_attr.get("name"),
                                "connected_nets": pin_nets,
                            }
                        else:
                            pin_entry = {
                                "pin_number": pin_attr.get("pin_number"),
                                "pin_name": pin_attr.get("name"),
                                "electrical_type": pin_attr.get("electrical_type"),
                                "connected_nets": "UNCONNECTED",
                            }
                    pins.append(pin_entry)

                if pins:
                    if is_passive:
                        pins.sort(key=lambda p: str(p.get("pin_name", "")))
                    else:
                        pins.sort(
                            key=lambda p: (
                                0,
                                int(str(p.get("pin_number"))),
                            )
                            if str(p.get("pin_number", "")).isdigit()
                            else (1, str(p.get("pin_number", "")))
                        )
                    comp_info["pins"] = pins

                if hidden_pin_count > 0:
                    comp_info["hidden_pins_count"] = hidden_pin_count

                if not is_passive and len(comp_pin_nodes) > 16:
                    sorted_candidates = sorted(fold_candidates, key=_pin_ref_sort_key)
                    comp_info["unused_pin_fold_suggestions"] = {
                        "mode": "manual",
                        "tool": "set_unused_pins_fold_tool",
                        "total_pins": len(comp_pin_nodes),
                        "connected_pins": connected_pin_count,
                        "hidden_pins": hidden_pin_count,
                        "candidate_count": len(sorted_candidates),
                        "suggested_hide_pins": sorted_candidates[:24],
                    }

                if not is_passive:
                    extras = _build_focus_layer_extras(
                        attr.get("part_number", ""),
                        include_ref_circuits=include_ref_circuits,
                    )
                    comp_info.update(extras)

            state["components"].append(comp_info)

    # 3) Net view.
    # Local nets: Belongs_To edge to the current module.
    # Global nets: scope=Global, visible in every module view.
    if include_nets:
        for node, attr in G.nodes(data=True):
            if attr.get("type") != "Net":
                continue

            is_local = G.has_edge(node, module_id) and G.get_edge_data(node, module_id).get("relation") == "Belongs_To"
            is_global = attr.get("scope") == "Global"
            if not (is_local or is_global):
                continue

            connected_pins = [n for n in G.neighbors(node) if G.nodes[n].get("type") == "Pin"]
            connected_vports = [n for n in G.neighbors(node) if G.nodes[n].get("type") == "VirtualPort"]

            net_entry = {
                "id": node,
                "net_type": attr.get("net_type", "Signal"),
                "connected_pins_summary": ", ".join(_pin_summary(G, p) for p in connected_pins),
            }

            scope = attr.get("scope", "Local")
            if scope != "Local":
                net_entry["scope"] = scope

            sem_role = attr.get("semantic_role", "").strip()
            if sem_role:
                net_entry["semantic_role"] = sem_role

            if connected_vports:
                net_entry["connected_virtual_ports"] = connected_vports

            state["nets"].append(net_entry)

    return yaml.dump(
        state,
        sort_keys=False,
        default_flow_style=False,
        Dumper=yaml.SafeDumper,
        allow_unicode=True,
    )