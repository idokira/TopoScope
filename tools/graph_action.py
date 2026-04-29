import inspect
import json
import re
from typing import Any, Dict, List, Optional, Union

import networkx as nx

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover
    class _FallbackArgsSchema:
        def __init__(self, fn):
            sig = inspect.signature(fn)
            props: Dict[str, Dict[str, Any]] = {}
            required: List[str] = []

            for name, param in sig.parameters.items():
                anno = param.annotation
                p_type = "string"
                if anno in {int, float}:
                    p_type = "number"
                elif anno is bool:
                    p_type = "boolean"
                elif anno in {dict, Dict}:
                    p_type = "object"
                elif anno in {list, List}:
                    p_type = "array"

                props[name] = {"type": p_type}
                if param.default is inspect._empty:
                    required.append(name)

            self._schema = {
                "type": "object",
                "properties": props,
                "required": required,
            }

        def schema(self) -> Dict[str, Any]:
            return self._schema

    class _FallbackTool:
        def __init__(self, fn):
            self._fn = fn
            self.name = fn.__name__
            self.description = (fn.__doc__ or "").strip()
            self.args_schema = _FallbackArgsSchema(fn)

        def invoke(self, args: Optional[Dict[str, Any]] = None):
            payload = args or {}
            return self._fn(**payload)

    def tool(fn):
        return _FallbackTool(fn)

from tools.build_graph import (
    add_component,
    add_passive_and_connect,
    add_virtual_port,
    connect_nodes,
    disconnect_nodes,
    create_global_net,
    create_net,
    remove_component,
    remove_net,
    request_new_vnet,
)
from tools.extract_state import inspect_design_overview, inspect_module_state


# ==========================================
# State management
# ==========================================
# Legacy compatibility: keep the default graph context.
# Upgrade: support session-level graph_id to avoid graph cross-talk under concurrent sessions.
GRAPH_CONTEXTS: Dict[str, nx.Graph] = {"default": nx.Graph()}
ACTIVE_GRAPH_ID = "default"
ACTIVE_MODULE_SCOPE: Optional[str] = None

# graph_id -> module_id -> focused component refdes
MODULE_FOCUS_STATE: Dict[str, Dict[str, Optional[str]]] = {}


def _resolve_graph_id(graph_id: Optional[str]) -> str:
    return graph_id or ACTIVE_GRAPH_ID


def _ensure_graph(graph_id: Optional[str]) -> nx.Graph:
    """
    Get or initialize the target graph context.
    """
    resolved_id = _resolve_graph_id(graph_id)
    if resolved_id not in GRAPH_CONTEXTS:
        GRAPH_CONTEXTS[resolved_id] = nx.Graph()
    if resolved_id not in MODULE_FOCUS_STATE:
        MODULE_FOCUS_STATE[resolved_id] = {}
    return GRAPH_CONTEXTS[resolved_id]


def _ensure_focus_state(graph_id: Optional[str]) -> Dict[str, Optional[str]]:
    resolved_id = _resolve_graph_id(graph_id)
    if resolved_id not in MODULE_FOCUS_STATE:
        MODULE_FOCUS_STATE[resolved_id] = {}
    return MODULE_FOCUS_STATE[resolved_id]


def set_active_graph(graph_id: str) -> None:
    """
    Explicitly switch the active graph context.
    """
    global ACTIVE_GRAPH_ID
    _ensure_graph(graph_id)
    ACTIVE_GRAPH_ID = graph_id


def set_current_graph(G: nx.Graph, graph_id: str = "default"):
    """
    External interface: inject the skeleton graph before the agent loop starts.
    """
    global ACTIVE_GRAPH_ID, ACTIVE_MODULE_SCOPE
    GRAPH_CONTEXTS[graph_id] = G
    ACTIVE_GRAPH_ID = graph_id
    MODULE_FOCUS_STATE[graph_id] = {}
    ACTIVE_MODULE_SCOPE = None


def get_current_graph(graph_id: Optional[str] = None) -> nx.Graph:
    """
    External interface: retrieve the final graph after the agent loop ends.
    """
    return _ensure_graph(graph_id)


def set_active_module_scope(module_id: Optional[str]) -> None:
    """
    Set the module scope that is currently editable in the design stage.
    """
    global ACTIVE_MODULE_SCOPE
    ACTIVE_MODULE_SCOPE = module_id


def clear_active_module_scope() -> None:
    set_active_module_scope(None)


def _module_scope_error(module_id: str) -> str:
    if ACTIVE_MODULE_SCOPE and module_id != ACTIVE_MODULE_SCOPE:
        return (
            f"Error: Module scope violation. Active module is '{ACTIVE_MODULE_SCOPE}', "
            f"but requested '{module_id}'."
        )
    return ""


def _module_of_component(G: nx.Graph, comp_id: str) -> Optional[str]:
    if not G.has_node(comp_id):
        return None
    for nbr in G.neighbors(comp_id):
        if (
            G.nodes[nbr].get("type") == "Module"
            and G.get_edge_data(comp_id, nbr).get("relation") == "Belongs_To"
        ):
            return nbr
    return None


def _module_of_pin(G: nx.Graph, pin_id: str) -> Optional[str]:
    if not G.has_node(pin_id):
        return None
    for nbr in G.neighbors(pin_id):
        if (
            G.nodes[nbr].get("type") == "Component"
            and G.get_edge_data(pin_id, nbr).get("relation") == "Belongs_To"
        ):
            return _module_of_component(G, nbr)
    return None


def _module_of_virtual_port(G: nx.Graph, vp_id: str) -> Optional[str]:
    if not G.has_node(vp_id):
        return None
    for nbr in G.neighbors(vp_id):
        if (
            G.nodes[nbr].get("type") == "Module"
            and G.get_edge_data(vp_id, nbr).get("relation") == "Belongs_To"
        ):
            return nbr
    return None


def _module_of_net(G: nx.Graph, net_id: str) -> Optional[str]:
    if not G.has_node(net_id):
        return None
    if G.nodes[net_id].get("scope") == "Global":
        return None
    for nbr in G.neighbors(net_id):
        if (
            G.nodes[nbr].get("type") == "Module"
            and G.get_edge_data(net_id, nbr).get("relation") == "Belongs_To"
        ):
            return nbr
    return None


def _node_in_active_scope(G: nx.Graph, node_id: str) -> bool:
    if not ACTIVE_MODULE_SCOPE:
        return True

    if not G.has_node(node_id):
        return True

    n_type = G.nodes[node_id].get("type")
    if n_type == "Module":
        return node_id == ACTIVE_MODULE_SCOPE
    if n_type == "Component":
        return _module_of_component(G, node_id) == ACTIVE_MODULE_SCOPE
    if n_type == "Pin":
        return _module_of_pin(G, node_id) == ACTIVE_MODULE_SCOPE
    if n_type == "VirtualPort":
        return _module_of_virtual_port(G, node_id) == ACTIVE_MODULE_SCOPE
    if n_type == "Net":
        # Global nets are visible from any module.
        if G.nodes[node_id].get("scope") == "Global":
            return True
        return _module_of_net(G, node_id) == ACTIVE_MODULE_SCOPE

    return False


def _pin_nodes_of_component(G: nx.Graph, comp_id: str) -> List[str]:
    """
    Return all pin nodes under a component.
    """
    if not G.has_node(comp_id) or G.nodes[comp_id].get("type") != "Component":
        return []

    pins: List[str] = []
    for nbr in G.neighbors(comp_id):
        if (
            G.nodes[nbr].get("type") == "Pin"
            and G.get_edge_data(comp_id, nbr).get("relation") == "Belongs_To"
        ):
            pins.append(nbr)
    return pins


def _pin_has_net_connection(G: nx.Graph, pin_id: str) -> bool:
    """
    Check whether a pin is already connected to any Net through Connected_To.
    """
    if not G.has_node(pin_id) or G.nodes[pin_id].get("type") != "Pin":
        return False

    for nbr in G.neighbors(pin_id):
        if (
            G.nodes[nbr].get("type") == "Net"
            and G.get_edge_data(pin_id, nbr).get("relation") == "Connected_To"
        ):
            return True
    return False


def _pin_is_hidden_in_view(G: nx.Graph, pin_id: str) -> bool:
    if not G.has_node(pin_id) or G.nodes[pin_id].get("type") != "Pin":
        return False
    return bool(G.nodes[pin_id].get("hidden_in_view", False))


def _pin_should_keep_visible(G: nx.Graph, pin_id: str) -> bool:
    """
    When folding pins for high-pin-count components, keep critical pins visible by default so the model
    can still continue wiring after folding.
    """
    if not G.has_node(pin_id) or G.nodes[pin_id].get("type") != "Pin":
        return False

    attrs = G.nodes[pin_id]
    electrical = str(attrs.get("electrical_type", "")).strip().lower()
    if any(k in electrical for k in ["power", "ground", "supply"]):
        return True

    name = str(attrs.get("name", "")).strip().lower()
    keep_tokens = [
        "vdd", "vcc", "vss", "gnd", "agnd", "dgnd", "vbat", "vref", "vin", "vout",
        "rst", "reset", "nrst", "boot", "swd", "swclk", "swdio", "jtag", "osc", "xtal",
        "scl", "sda", "tx", "rx",
    ]
    return any(tok in name for tok in keep_tokens)


def _pin_sort_key(G: nx.Graph, pin_id: str) -> tuple:
    attrs = G.nodes[pin_id]
    pin_num = str(attrs.get("pin_number", "")).strip()
    if pin_num.isdigit():
        return (0, int(pin_num), pin_id)
    return (1, pin_num, pin_id)


def _resolve_component_pin_alias(G: nx.Graph, comp_id: str, pin_token: str) -> Optional[str]:
    """
    Resolve component pin aliases, supporting forms such as C1.1 / C1.pin_1 / C1:EN.
    """
    token = str(pin_token).strip()
    if not token:
        return None

    token_l = token.lower()
    numeric_match = re.search(r"(?:pin)?_?([0-9]+)$", token_l)
    if numeric_match:
        candidate = f"{comp_id}_Pin{numeric_match.group(1)}"
        if G.has_node(candidate) and G.nodes[candidate].get("type") == "Pin":
            return candidate

    token_norm = re.sub(r"[^a-z0-9]+", "", token_l)
    for pin_node in _pin_nodes_of_component(G, comp_id):
        attrs = G.nodes[pin_node]
        pin_num = str(attrs.get("pin_number", "")).strip().lower()
        pin_name = str(attrs.get("name", "")).strip().lower()

        if token_l in {pin_num, pin_name}:
            return pin_node

        if token_norm:
            pin_num_norm = re.sub(r"[^a-z0-9]+", "", pin_num)
            pin_name_norm = re.sub(r"[^a-z0-9]+", "", pin_name)
            if token_norm in {pin_num_norm, pin_name_norm}:
                return pin_node

    return None


def _resolve_node_alias(G: nx.Graph, node_id: str) -> str:
    """
    Try to resolve common model-produced aliases into real node IDs.
    """
    raw = str(node_id).strip()
    if not raw:
        return raw

    if G.has_node(raw):
        return raw

    # Common forms: C1.1 / U3:EN.
    match = re.match(r"^([A-Za-z][A-Za-z0-9]*)[.:]([A-Za-z0-9_+-]+)$", raw)
    if match:
        comp_id, pin_token = match.group(1), match.group(2)
        if G.has_node(comp_id) and G.nodes[comp_id].get("type") == "Component":
            resolved = _resolve_component_pin_alias(G, comp_id, pin_token)
            if resolved:
                return resolved

    # Support legacy or guessed forms such as C1_pin_1 / C1_1.
    match = re.match(r"^([A-Za-z][A-Za-z0-9]*)_(?:pin_?)?([0-9]+)$", raw, flags=re.IGNORECASE)
    if match:
        comp_id, pin_num = match.group(1), match.group(2)
        if G.has_node(comp_id) and G.nodes[comp_id].get("type") == "Component":
            candidate = f"{comp_id}_Pin{pin_num}"
            if G.has_node(candidate) and G.nodes[candidate].get("type") == "Pin":
                return candidate

    return raw


def _coerce_bool(value: Any, default: bool) -> bool:
    """
    Support cases where the model serializes boolean values as strings, such as "True" / "false" / "1".
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return default


def _coerce_int(value: Any, default: int, minimum: int = 0) -> int:
    """
    Support cases where the model serializes numbers as strings, such as "16".
    """
    try:
        out = int(value)
    except Exception:
        try:
            out = int(float(str(value).strip()))
        except Exception:
            out = default
    return max(minimum, out)


def _coerce_str_list(value: Optional[Union[List[str], str]]) -> List[str]:
    """
    Support both list input and JSON-string input, such as ["U2.1"] or "[\\"U2.1\\"]".
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    text = str(value).strip()
    if not text:
        return []

    # Prefer parsing JSON array strings first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass

    # Fallback: comma-separated text or simple bracketed string.
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [p.strip().strip('"').strip("'") for p in text.split(",")]
    return [p for p in parts if p]


def _set_module_focus(graph_id: Optional[str], module_id: str, refdes: Optional[str]) -> None:
    state = _ensure_focus_state(graph_id)
    state[module_id] = refdes


def _get_module_focus(graph_id: Optional[str], module_id: str) -> Optional[str]:
    state = _ensure_focus_state(graph_id)
    return state.get(module_id)