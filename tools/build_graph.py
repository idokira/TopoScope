import json
import os
import re
from typing import Any, Dict, List, Optional, Set

import networkx as nx


# ==========================================
# 常量配置
# ==========================================
COMPONENT_DB_PATHS = [
    "benchmark/component_data/Extracted_Components.json",
    "component_data/Extracted_Components.json",
    "../component_data/Extracted_Components.json",
    "Extracted_Components.json",
    "tools/Extracted_Components.json",
    "../Extracted_Components.json",
]

_COMPONENT_DB_CACHE: Optional[List[Dict[str, Any]]] = None

_PASSIVE_TYPE_CONFIG: Dict[str, Dict[str, Any]] = {
    "Resistor": {
        "refdes_prefix": "R",
        "part_number_prefix": "RES",
        "part_number_example": "RES_10K",
        "pin_names": ["1", "2"],
        "is_passive": True,
    },
    "Capacitor": {
        "refdes_prefix": "C",
        "part_number_prefix": "CAP",
        "part_number_example": "CAP_10uF",
        "pin_names": ["1", "2"],
        "is_passive": True,
    },
    "Inductor": {
        "refdes_prefix": "L",
        "part_number_prefix": "IND",
        "part_number_example": "IND_4.7uH",
        "pin_names": ["1", "2"],
        "is_passive": True,
    },
    "Diode": {
        "refdes_prefix": "D",
        "part_number_prefix": "DIODE",
        "part_number_example": "DIODE_SS34",
        "pin_names": ["A", "K"],
        "is_passive": True,
    },
    "LED": {
        "refdes_prefix": "D",
        "part_number_prefix": "LED",
        "part_number_example": "LED_RED",
        "pin_names": ["A", "K"],
        "is_passive": True,
    },
    "Crystal": {
        "refdes_prefix": "Y",
        "part_number_prefix": "CRYSTAL",
        "part_number_example": "CRYSTAL_8MHz",
        "pin_names": ["1", "2"],
        "is_passive": True,
    },
    "Resonator": {
        # 与 Crystal 共享命名规则，兼容器件库里常见的 Resonator 类型。
        "refdes_prefix": "Y",
        "part_number_prefix": "CRYSTAL",
        "part_number_example": "CRYSTAL_8MHz",
        "pin_names": ["1", "2"],
        "is_passive": True,
    },
    "Fuse": {
        "refdes_prefix": "F",
        "part_number_prefix": "FUSE",
        "part_number_example": "FUSE_1A",
        "pin_names": ["1", "2"],
        "is_passive": True,
    },
    "BJT_NPN": {
        "refdes_prefix": "Q",
        "part_number_prefix": "BJT_NPN",
        "part_number_example": "BJT_NPN_2N3904",
        "pin_names": ["B", "C", "E"],
        "is_passive": False,
    },
    "BJT_PNP": {
        "refdes_prefix": "Q",
        "part_number_prefix": "BJT_PNP",
        "part_number_example": "BJT_PNP_2N3906",
        "pin_names": ["B", "C", "E"],
        "is_passive": False,
    },
    "MOSFET_N": {
        "refdes_prefix": "Q",
        "part_number_prefix": "MOSFET_N",
        "part_number_example": "MOSFET_N_2N7002",
        "pin_names": ["G", "D", "S"],
        "is_passive": False,
    },
    "MOSFET_P": {
        "refdes_prefix": "Q",
        "part_number_prefix": "MOSFET_P",
        "part_number_example": "MOSFET_P_AO3401A",
        "pin_names": ["G", "D", "S"],
        "is_passive": False,
    },
}


# ==========================================
# 辅助函数：数据库与字符串解析
# ==========================================
def _find_component_db_path() -> Optional[str]:
    """
    兼容不同运行目录，定位器件数据库文件。
    """
    for path in COMPONENT_DB_PATHS:
        if os.path.exists(path):
            return path
    return None


def _load_component_db() -> List[Dict[str, Any]]:
    """
    读取并缓存器件数据库，避免高频重复 IO。
    """
    global _COMPONENT_DB_CACHE

    if _COMPONENT_DB_CACHE is not None:
        return _COMPONENT_DB_CACHE

    db_path = _find_component_db_path()
    if not db_path:
        _COMPONENT_DB_CACHE = []
        return _COMPONENT_DB_CACHE

    try:
        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _COMPONENT_DB_CACHE = data if isinstance(data, list) else []
    except Exception as e:
        print(f"Error reading component database: {e}")
        _COMPONENT_DB_CACHE = []

    return _COMPONENT_DB_CACHE


def get_component_data(part_number: str) -> Optional[Dict[str, Any]]:
    """
    根据型号从本地 JSON 库中检索器件数据。

    规则：
    1) 优先精确匹配 part_number；
    2) 然后精确匹配 component 字段；
    3) 精确失败后再降级为包含匹配 part_number；
    4) 最后包含匹配 component 字段。
    """
    data = _load_component_db()
    if not data:
        return None

    query_lower = part_number.strip().lower()
    if not query_lower:
        return None

    # 1) 精确匹配 part_number
    for item in data:
        if item.get("part_number", "").strip().lower() == query_lower:
            return item

    # 2) 精确匹配 component 字段
    for item in data:
        if item.get("component", "").strip().lower() == query_lower:
            return item

    # 3) 包含匹配 part_number
    for item in data:
        if query_lower in item.get("part_number", "").lower():
            return item

    # 4) 包含匹配 component 字段
    for item in data:
        if query_lower in item.get("component", "").lower():
            return item

    return None


def _parse_voltage_value(raw: Any) -> Optional[float]:
    """
    将电压文本解析为浮点值（单位: V）。
    支持: 3.3, "3.3V", "=5V", "12"。
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)

    text = str(raw).strip().lower()
    match = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
    if not match:
        return None

    try:
        return float(match.group(1))
    except Exception:
        return None


def _normalize_virtual_electrical_type(raw_type: Any) -> str:
    """
    将虚拟端口电气类型归一化为 power / ground / signal。
    """
    text = str(raw_type or "").strip().lower()
    if "ground" in text or text == "gnd":
        return "ground"
    if "power" in text or "vcc" in text or "vdd" in text or "vin" in text or "vout" in text:
        return "power"
    return "signal"


def _safe_python_var(name: str) -> str:
    """
    将任意字符串转换成可用的 Python 变量名。
    """
    var = re.sub(r"\W+", "_", str(name))
    if not var:
        var = "NET"
    if var[0].isdigit():
        var = f"N_{var}"
    return var


def _normalize_passive_value_token(value: str) -> str:
    """
    将 value 标准化为 Part Number 的 VALUE token（去掉空白）。
    """
    return re.sub(r"\s+", "", value.strip())


def _canonicalize_passive_type(part_type: str) -> Optional[str]:
    """
    归一化动态器件类型，兼容大小写与常见别名。
    """
    token = str(part_type or "").strip()
    if not token:
        return None

    alias_map = {
        "resistor": "Resistor",
        "capacitor": "Capacitor",
        "inductor": "Inductor",
        "diode": "Diode",
        "led": "LED",
        "crystal": "Crystal",
        "xtal": "Crystal",
        "resonator": "Resonator",
        "fuse": "Fuse",
        "bjt_npn": "BJT_NPN",
        "npn": "BJT_NPN",
        "transistor_npn": "BJT_NPN",
        "bjt_pnp": "BJT_PNP",
        "pnp": "BJT_PNP",
        "transistor_pnp": "BJT_PNP",
        "mosfet_n": "MOSFET_N",
        "nmos": "MOSFET_N",
        "n_mos": "MOSFET_N",
        "mosfet_p": "MOSFET_P",
        "pmos": "MOSFET_P",
        "p_mos": "MOSFET_P",
    }

    canonical = alias_map.get(token.lower())
    if canonical and canonical in _PASSIVE_TYPE_CONFIG:
        return canonical

    if token in _PASSIVE_TYPE_CONFIG:
        return token
    return None


def _validate_passive_refdes(refdes: str, part_type: str) -> Optional[str]:
    """
    校验动态器件 Refdes 是否符合命名规则（如 R1/C1/L1/D1/Q1）。
    """
    cfg = _PASSIVE_TYPE_CONFIG.get(part_type)
    if not cfg:
        allowed = sorted(_PASSIVE_TYPE_CONFIG.keys())
        return f"Error: Unsupported dynamic component type '{part_type}'. Allowed types: {allowed}."

    expected_prefix = cfg["refdes_prefix"]
    if not re.fullmatch(rf"{expected_prefix}\d+", refdes):
        return (
            f"Error: Refdes '{refdes}' is invalid for type '{part_type}'. "
            f"Expected format '{expected_prefix}<number>' (e.g., {expected_prefix}1)."
        )

    return None


def _validate_passive_part_number(part_number: str, part_type: str, value: str) -> Optional[str]:
    """
    校验动态器件 Part Number 是否符合命名规则（如 CAP_10uF、BJT_NPN_2N3904、MOSFET_N_2N7002）。
    """
    cfg = _PASSIVE_TYPE_CONFIG.get(part_type)
    if not cfg:
        allowed = sorted(_PASSIVE_TYPE_CONFIG.keys())
        return f"Error: Unsupported dynamic component type '{part_type}'. Allowed types: {allowed}."

    token = part_number.strip()
    if not token:
        return f"Error: part_number is required for component type '{part_type}'."

    pn_prefix = cfg["part_number_prefix"]
    if not re.fullmatch(rf"{pn_prefix}_[A-Za-z0-9.+%-]+(?:_[A-Za-z0-9.+%-]+)*", token):
        pn_example = cfg["part_number_example"]
        return (
            f"Error: part_number '{part_number}' is invalid for type '{part_type}'. "
            f"Expected format '{pn_prefix}_<VALUE>' or '{pn_prefix}_<VALUE>_<EXTRA>' "
            f"(e.g., {pn_example})."
        )

    normalized_value = _normalize_passive_value_token(value)
    if normalized_value and not token.startswith(f"{pn_prefix}_{normalized_value}"):
        return (
            f"Error: part_number '{part_number}' does not match value '{value}'. "
            f"Expected prefix '{pn_prefix}_{normalized_value}'."
        )

    return None


# ==========================================
# 图构建与图操作
# ==========================================
def _add_planned_global_nets(G: nx.Graph, json_data: Dict[str, Any]) -> None:
    """
    根据 Planner 输出的 initial_global_nets 初始化全局网络。

    设计意图：
    - 不再硬编码默认公共网络，避免引入与任务无关的冗余网络。
    - 初始全局网络由 Planner 显式给出，后续仍可通过工具动态新增。
    """
    planned_nets = json_data.get("initial_global_nets", [])
    if not isinstance(planned_nets, list):
        print("Warning: initial_global_nets should be a list. Skip initialization of planned global nets.")
        return

    for idx, net in enumerate(planned_nets):
        if not isinstance(net, dict):
            print(f"Warning: initial_global_nets[{idx}] is not an object. Skipped.")
            continue

        net_name = str(net.get("net_name", net.get("name", ""))).strip()
        if not net_name:
            print(f"Warning: initial_global_nets[{idx}] missing net_name/name. Skipped.")
            continue

        create_global_net(
            G,
            net_name=net_name,
            net_type=str(net.get("net_type", "Power")),
            semantic_role=str(net.get("semantic_role", "Planned initial global net")),
            source_pin=str(net.get("source_pin", "")),
        )


def _is_phase0_v4_schema(json_data: Dict[str, Any]) -> bool:
    """
    检测是否为 v4 Phase 0 结构：
    - modules[].components（refdes 列表）
    - macro_connections[].source/target/net_name
    """
    modules = json_data.get("modules", [])
    has_components_field = any(isinstance(m, dict) and "components" in m for m in modules)

    macro_connections = json_data.get("macro_connections", [])
    has_v4_macro = any(
        isinstance(conn, dict) and (
            "source" in conn
            or "target" in conn
            or "net_name" in conn
        )
        for conn in macro_connections
    )

    return bool(has_components_field or has_v4_macro)


def _build_component_lookup(component_catalog: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    基于输入 component_list 建立 refdes -> component meta 映射。
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    if not isinstance(component_catalog, list):
        return lookup

    for item in component_catalog:
        if not isinstance(item, dict):
            continue
        refdes = str(item.get("refdes", "")).strip()
        if not refdes:
            continue
        lookup[refdes] = item

    return lookup


def _normalize_module_component_specs(
    module: Dict[str, Any],
    component_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    兼容两种 Phase 0 模块字段：
    - v4: components: ["U1", "U2"]
    - legacy: main_components: [{"refdes": "U1", "part_number": "..."}]
    """
    normalized: List[Dict[str, str]] = []
    seen_refdes: Set[str] = set()

    raw_components = module.get("components", [])
    if isinstance(raw_components, list) and raw_components:
        for item in raw_components:
            refdes = ""
            part_number = ""
            description = ""

            if isinstance(item, dict):
                refdes = str(item.get("refdes", "")).strip()
                part_number = str(item.get("part_number", "")).strip()
                description = str(item.get("description", "")).strip()
            else:
                refdes = str(item).strip()

            if not refdes or refdes in seen_refdes:
                continue

            lookup_item = component_lookup.get(refdes, {})
            if not part_number:
                part_number = str(lookup_item.get("part_number", "")).strip()
            if not description:
                description = str(lookup_item.get("description", "")).strip()

            normalized.append(
                {
                    "refdes": refdes,
                    "part_number": part_number,
                    "description": description,
                }
            )
            seen_refdes.add(refdes)

        return normalized

    # 旧版字段兼容
    legacy_components = module.get("main_components", [])
    if isinstance(legacy_components, list):
        for item in legacy_components:
            if not isinstance(item, dict):
                continue

            refdes = str(item.get("refdes", "")).strip()
            if not refdes or refdes in seen_refdes:
                continue

            normalized.append(
                {
                    "refdes": refdes,
                    "part_number": str(item.get("part_number", "")).strip(),
                    "description": str(item.get("description", "")).strip(),
                }
            )
            seen_refdes.add(refdes)

    return normalized


def _module_from_endpoint(endpoint: str) -> str:
    token = str(endpoint or "").strip()
    if "." in token:
        return token.split(".", 1)[0].strip()
    return ""


def _port_from_endpoint(endpoint: str) -> str:
    token = str(endpoint or "").strip()
    if "." in token:
        return token.split(".", 1)[1].strip()
    return ""


def _normalize_macro_connection(conn: Dict[str, Any], index: int) -> Dict[str, str]:
    """
    兼容新旧宏观连线字段并归一化为 source/target/net_name。
    """
    source = str(conn.get("source", conn.get("source_port", ""))).strip()
    target = str(conn.get("target", conn.get("target_port", ""))).strip()

    net_name = str(conn.get("net_name", "")).strip()
    if not net_name:
        net_name = _port_from_endpoint(source) or _port_from_endpoint(target)
    if not net_name:
        net_name = f"AUTO_NET_{index + 1}"

    return {
        "source": source,
        "target": target,
        "net_name": net_name,
        "description": str(conn.get("description", "")).strip(),
    }


def _derive_shared_nets_from_plan(
    json_data: Dict[str, Any],
    module_ids: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """
    从 Phase 0 规划中推导 shared_nets：
    - initial_global_nets: 对所有模块可见
    - macro_connections: 对 source/target 所在模块可见
    """
    shared_nets: Dict[str, Dict[str, Any]] = {}

    for net in json_data.get("initial_global_nets", []):
        if not isinstance(net, dict):
            continue

        net_name = str(net.get("net_name", net.get("name", ""))).strip()
        if not net_name:
            continue

        shared_nets[net_name] = {
            "type": str(net.get("net_type", "Power")),
            "modules": set(module_ids),
        }

    for idx, conn in enumerate(json_data.get("macro_connections", [])):
        if not isinstance(conn, dict):
            continue

        normalized = _normalize_macro_connection(conn, idx)
        net_name = normalized["net_name"]
        src_mod = _module_from_endpoint(normalized["source"])
        tgt_mod = _module_from_endpoint(normalized["target"])

        if net_name not in shared_nets:
            net_type = "Signal"
            net_lower = net_name.lower()
            if any(x in net_lower for x in ["vcc", "vdd", "vin", "vout", "power", "3v3", "5v", "3.3v", "5.0v"]):
                net_type = "Power"
            elif any(x in net_lower for x in ["gnd", "vss", "ground"]):
                net_type = "Ground"
            shared_nets[net_name] = {
                "type": net_type,
                "modules": set(),
            }

        if src_mod in module_ids:
            shared_nets[net_name]["modules"].add(src_mod)
        if tgt_mod in module_ids:
            shared_nets[net_name]["modules"].add(tgt_mod)

    return shared_nets


def build_initial_skeleton_graph(
    json_data: Dict[str, Any],
    component_catalog: Optional[List[Dict[str, Any]]] = None,
) -> nx.Graph:
    """
    解析 Phase 0 的宏观 JSON，构建初始 NetworkX 图。

    关键行为：
    1) 建立 Module / Component / Pin 的基础骨架。
    2) 自动实例化器件引脚（由数据库驱动）。
    3) 按 Planner 输出初始化初始全局网络。
    4) v4 Phase 0 下，从 macro_connections 推导 shared_nets 并创建共享信号网络。

    兼容性：
    - 若检测到旧版 schema（main_components + source_port/target_port），保留 VirtualPort 入图行为。
    """
    G = nx.Graph()
    component_lookup = _build_component_lookup(component_catalog)
    phase0_v4 = _is_phase0_v4_schema(json_data)

    # 初始化由 Planner 指定的初始全局网络层。
    _add_planned_global_nets(G, json_data)

    module_ids: Set[str] = set()

    # 第一阶段：建立模块与内部实体。
    for module in json_data.get("modules", []):
        module_id = module["module_id"]
        module_ids.add(module_id)

        G.add_node(
            module_id,
            type="Module",
            name=module_id,
            description=module.get("description", ""),
        )

        # v4 约束：VirtualPort 仅用于 Phase0 定义接口，不进入图。
        # 旧版兼容：仍允许 VirtualPort 入图。
        if not phase0_v4:
            for vp in module.get("virtual_ports", []):
                vp_id = f"{module_id}_{vp['port_name']}"
                G.add_node(
                    vp_id,
                    type="VirtualPort",
                    name=vp["port_name"],
                    electrical_type=vp.get("electrical_type", "Signal"),
                    direction=vp.get("direction", "Bi-dir"),
                    expected_voltage=vp.get("expected_voltage"),
                    expected_type=vp.get("expected_type", vp.get("electrical_type", "Signal")),
                )
                G.add_edge(vp_id, module_id, relation="Belongs_To")

        # 挂载核心器件与引脚实例化 (Pin -> Belongs_To -> Component -> Belongs_To -> Module)
        for comp in _normalize_module_component_specs(module, component_lookup):
            refdes = comp["refdes"]
            part_number = str(comp.get("part_number", "")).strip()

            if not G.has_node(refdes):
                G.add_node(
                    refdes,
                    type="Component",
                    refdes=refdes,
                    part_number=part_number,
                    description=comp.get("description", ""),
                )
            if not G.has_edge(refdes, module_id):
                G.add_edge(refdes, module_id, relation="Belongs_To")

            if not part_number:
                print(f"Warning: Missing part_number for component {refdes}. Pin instantiation skipped.")
                continue

            lib_data = get_component_data(part_number)
            if lib_data:
                for pin in lib_data.get("pins", []):
                    pin_id = f"{refdes}_Pin{pin['number']}"
                    if G.has_node(pin_id):
                        continue
                    elec_type = pin.get("electrical_type") or pin.get("type", "Passive")
                    G.add_node(
                        pin_id,
                        type="Pin",
                        name=pin.get("name", str(pin["number"])),
                        pin_number=pin["number"],
                        electrical_type=elec_type,
                        description=pin.get("description", ""),
                    )
                    G.add_edge(pin_id, refdes, relation="Belongs_To")
            else:
                print(f"Warning: Part number {part_number} not found in library.")

    if phase0_v4:
        shared_nets = _derive_shared_nets_from_plan(json_data, module_ids)

        for net_name, meta in shared_nets.items():
            net_type = str(meta.get("type", "Signal") or "Signal")
            create_global_net(
                G,
                net_name=net_name,
                net_type=net_type,
                net_class=net_type,
                semantic_role="Phase0 shared net",
                source_pin="",
            )
            if G.has_node(net_name):
                G.nodes[net_name]["shared_modules"] = sorted(meta.get("modules", set()))

        G.graph["shared_nets"] = {
            net_name: {
                "type": str(meta.get("type", "Signal") or "Signal"),
                "modules": sorted(meta.get("modules", set())),
            }
            for net_name, meta in sorted(shared_nets.items())
        }
    else:
        # 旧版：建立跨模块的宏观连接意图 (VirtualPort -> Connected_To -> VirtualPort)
        for conn in json_data.get("macro_connections", []):
            source_id = conn["source_port"].replace(".", "_")
            target_id = conn["target_port"].replace(".", "_")

            if G.has_node(source_id) and G.has_node(target_id):
                G.add_edge(
                    source_id,
                    target_id,
                    relation="Connected_To",
                    description=conn.get("description", ""),
                )
            else:
                print(f"Error: Cannot connect {source_id} and {target_id}, nodes missing.")

    return G


def erc_probe_validate(G: nx.Graph, pin_id: str, net_id: str) -> tuple[bool, str]:
    """
    图论探针：在 Pin 接入 Net 前执行即时 ERC 冲突检查。
    返回 (是否通过, 信息)。
    """
    def _normalize_category(raw_type: Any) -> str:
        text = str(raw_type or "").strip().lower().replace("_", "").replace("-", "")
        if not text:
            return "unknown"

        if text in {"nc", "dnc"}:
            return "no_connect"
        if text in {"ground", "gnd"}:
            return "ground"
        if text in {"power", "powerin", "powerout"}:
            return "power"
        if text in {"output", "analogout", "tristate"}:
            return "strong_driver"
        if text in {"opendrain"}:
            return "open_drain"
        if text in {"input", "analogin"}:
            return "input"
        if text in {"bidir", "analogbi"}:
            return "bidir"
        if text in {"passive"}:
            return "passive"
        return "signal"

    new_pin_type = G.nodes[pin_id].get("electrical_type")
    if not new_pin_type:
        return False, f"ERC Error: Pin '{pin_id}' lacks an electrical_type attribute."

    existing_pin_types: List[Any] = []
    for neighbor in G.neighbors(net_id):
        n_type = G.nodes[neighbor].get("type")
        if n_type in {"Pin", "VirtualPort"}:
            existing_pin_types.append(G.nodes[neighbor].get("electrical_type"))

    simulated_categories = [_normalize_category(t) for t in (existing_pin_types + [new_pin_type])]
    category_counts: Dict[str, int] = {}
    for category in simulated_categories:
        category_counts[category] = category_counts.get(category, 0) + 1

    # 规则 0: NC/DNC 不允许被连接。
    if category_counts.get("no_connect", 0) > 0:
        return False, f"ERC ERROR: Net '{net_id}' contains NC/DNC pin(s), which must remain unconnected."

    # 规则 1: Power + Ground 直短。
    if category_counts.get("power", 0) > 0 and category_counts.get("ground", 0) > 0:
        return False, f"ERC FATAL: Attempting to short Power and Ground on '{net_id}'."

    # 规则 2: 强驱动冲突（如 Output/AnalogOut/TriState）。
    if category_counts.get("strong_driver", 0) > 1:
        return False, f"ERC ERROR: Multiple Output pins detected on '{net_id}'. This causes logic contention."

    # 规则 3: 强驱动端直接连到 Power/Ground。
    if category_counts.get("strong_driver", 0) > 0 and (
        category_counts.get("power", 0) > 0 or category_counts.get("ground", 0) > 0
    ):
        return False, f"ERC ERROR: Cannot connect an Output pin directly to Power or Ground on '{net_id}'."

    return True, "ERC Passed."


def create_net(
    G: nx.Graph,
    module_id: str,
    net_name: str,
    net_type: str = "Signal",
    net_class: Optional[str] = None,
    semantic_role: str = "",
    source_pin: str = "",
) -> str:
    """
    在指定模块内创建局部网络节点，支持语义属性。
    """
    if not G.has_node(module_id):
        return f"Error: Module '{module_id}' does not exist."

    net_id = f"{module_id}_{net_name}"
    if G.has_node(net_id):
        return f"Error: Net '{net_name}' already exists in {module_id}."

    G.add_node(
        net_id,
        type="Net",
        name=net_name,
        scope="Local",
        net_type=net_type,
        net_class=net_class or net_type,
        semantic_role=semantic_role,
        source_pin=source_pin,
    )
    G.add_edge(net_id, module_id, relation="Belongs_To")

    return f"Success: Net '{net_id}' created."


def create_global_net(
    G: nx.Graph,
    net_name: str,
    net_type: str = "Power",
    net_class: Optional[str] = None,
    semantic_role: str = "",
    source_pin: str = "",
) -> str:
    """
    创建全局网络节点（不属于某个模块）。
    """
    if G.has_node(net_name):
        if G.nodes[net_name].get("type") != "Net":
            return f"Error: Node '{net_name}' exists and is not a Net."
        return f"Success: Global Net '{net_name}' already exists."

    G.add_node(
        net_name,
        type="Net",
        name=net_name,
        scope="Global",
        net_type=net_type,
        net_class=net_class or net_type,
        semantic_role=semantic_role,
        source_pin=source_pin,
    )
    return f"Success: Global Net '{net_name}' created."


def add_virtual_port(
    G: nx.Graph,
    module_id: str,
    port_name: str,
    electrical_type: str,
    direction: str = "Bi-dir",
    expected_voltage: Optional[str] = None,
    expected_type: Optional[str] = None,
) -> str:
    """
    在指定模块上增加虚拟端口（VNet脚手架）。
    """
    if not G.has_node(module_id):
        return f"Error: Module '{module_id}' does not exist."

    vp_id = f"{module_id}_{port_name}"
    if G.has_node(vp_id):
        return f"Error: Virtual port '{vp_id}' already exists."

    G.add_node(
        vp_id,
        type="VirtualPort",
        name=port_name,
        electrical_type=electrical_type,
        direction=direction,
        expected_voltage=expected_voltage,
        expected_type=expected_type or electrical_type,
    )
    G.add_edge(vp_id, module_id, relation="Belongs_To")
    return f"Success: Virtual port '{vp_id}' created."


def request_new_vnet(
    G: nx.Graph,
    source_module: str,
    target_module: str,
    vnet_spec: Dict[str, Any],
) -> str:
    """
    跨模块动态协商接口：在两个模块上新增同名虚拟端口并建立连接。
    """
    port_name = vnet_spec.get("name")
    if not port_name:
        return "Error: vnet_spec.name is required."

    e_type = vnet_spec.get("type", vnet_spec.get("electrical_type", "Signal"))
    voltage = vnet_spec.get("voltage", vnet_spec.get("expected_voltage"))
    expected_type = vnet_spec.get("expected_type", e_type)

    source_direction = vnet_spec.get("source_direction", "Output")
    target_direction = vnet_spec.get("target_direction", "Input")

    # 如果端口已存在则复用，不存在则创建。
    src_vp_id = f"{source_module}_{port_name}"
    if not G.has_node(src_vp_id):
        msg = add_virtual_port(
            G,
            source_module,
            port_name,
            electrical_type=e_type,
            direction=source_direction,
            expected_voltage=voltage,
            expected_type=expected_type,
        )
        if msg.startswith("Error"):
            return msg

    tgt_vp_id = f"{target_module}_{port_name}"
    if not G.has_node(tgt_vp_id):
        msg = add_virtual_port(
            G,
            target_module,
            port_name,
            electrical_type=e_type,
            direction=target_direction,
            expected_voltage=voltage,
            expected_type=expected_type,
        )
        if msg.startswith("Error"):
            return msg

    if not G.has_edge(src_vp_id, tgt_vp_id):
        G.add_edge(src_vp_id, tgt_vp_id, relation="Connected_To", description="Dynamic negotiated VNet")

    return f"Success: Dynamic VNet '{port_name}' established between {source_module} and {target_module}."


def connect_nodes(G: nx.Graph, node1_id: str, node2_id: str) -> str:
    """
    建立两个节点之间的电气连接，并执行拓扑/ERC检查。

    注意：
    - 按用户评论保留“Pin 可连接多个 Net”的能力，不在此处强制 S4。
    """
    if not G.has_node(node1_id):
        return f"Error: Node '{node1_id}' does not exist."
    if not G.has_node(node2_id):
        return f"Error: Node '{node2_id}' does not exist."

    type1 = G.nodes[node1_id].get("type")
    type2 = G.nodes[node2_id].get("type")

    # 规则 A：禁止 Pin 直接连 Pin。
    if type1 == "Pin" and type2 == "Pin":
        return (
            "Error: Topological violation. A Pin cannot connect directly to another Pin. "
            "You must create a Net first, and connect both Pins to that Net."
        )

    # 规则 B：禁止 Net 直接连 Net。
    if type1 == "Net" and type2 == "Net":
        return "Error: Topological violation. Cannot connect Net to Net directly."

    # 合法连接：Pin-Net, Net-VirtualPort。
    valid_connections = [{"Pin", "Net"}, {"Net", "VirtualPort"}]
    current_connection = {type1, type2}
    if current_connection not in valid_connections:
        return f"Error: Invalid connection between {type1} and {type2}."

    if G.has_edge(node1_id, node2_id):
        return f"Success: {node1_id} and {node2_id} are already connected."

    pin_node = node1_id if type1 == "Pin" else node2_id
    net_node = node2_id if type2 == "Net" else node1_id

    if "Pin" in current_connection and "Net" in current_connection:
        is_valid, msg = erc_probe_validate(G, pin_node, net_node)
        if not is_valid:
            return msg
    else:
        msg = "No ERC required for this connection type."

    G.add_edge(node1_id, node2_id, relation="Connected_To")
    return f"Success: Connected {node1_id} to {node2_id}. {msg}"


def disconnect_nodes(G: nx.Graph, node1_id: str, node2_id: str) -> str:
    """
    断开两个节点之间的电气连接（仅允许断开 Connected_To）。
    """
    if not G.has_node(node1_id):
        return f"Error: Node '{node1_id}' does not exist."
    if not G.has_node(node2_id):
        return f"Error: Node '{node2_id}' does not exist."

    if not G.has_edge(node1_id, node2_id):
        return f"Success: {node1_id} and {node2_id} are already disconnected."

    relation = G.get_edge_data(node1_id, node2_id).get("relation")
    if relation != "Connected_To":
        return (
            f"Error: Only Connected_To edges can be removed via disconnect_nodes. "
            f"Current relation is '{relation}'."
        )

    G.remove_edge(node1_id, node2_id)
    return f"Success: Disconnected {node1_id} from {node2_id}."


def remove_component(G: nx.Graph, refdes: str, remove_orphan_nets: bool = True) -> str:
    """
    删除器件及其引脚。

    参数：
    - remove_orphan_nets: 是否自动清理因该器件删除而变成孤立的局部网络。
    """
    if not G.has_node(refdes):
        return f"Error: Component '{refdes}' does not exist."
    if G.nodes[refdes].get("type") != "Component":
        return f"Error: Node '{refdes}' is not a Component."

    pins = [
        n
        for n in G.neighbors(refdes)
        if G.nodes[n].get("type") == "Pin"
        and G.get_edge_data(refdes, n).get("relation") == "Belongs_To"
    ]

    touched_nets: Set[str] = set()
    for pin in pins:
        for nbr in G.neighbors(pin):
            if (
                G.nodes[nbr].get("type") == "Net"
                and G.get_edge_data(pin, nbr).get("relation") == "Connected_To"
            ):
                touched_nets.add(nbr)

    for pin in pins:
        if G.has_node(pin):
            G.remove_node(pin)
    if G.has_node(refdes):
        G.remove_node(refdes)

    removed_nets: List[str] = []
    if remove_orphan_nets:
        for net in sorted(touched_nets):
            if not G.has_node(net):
                continue
            if G.nodes[net].get("type") != "Net":
                continue
            if G.nodes[net].get("scope") == "Global":
                # 全局网络默认保留，不自动删除。
                continue

            connected_pins = [
                n
                for n in G.neighbors(net)
                if G.nodes[n].get("type") == "Pin"
                and G.get_edge_data(net, n).get("relation") == "Connected_To"
            ]
            connected_vports = [
                n
                for n in G.neighbors(net)
                if G.nodes[n].get("type") == "VirtualPort"
                and G.get_edge_data(net, n).get("relation") == "Connected_To"
            ]
            if connected_pins or connected_vports:
                continue

            G.remove_node(net)
            removed_nets.append(net)

    msg = f"Success: Component '{refdes}' and {len(pins)} pin(s) removed."
    if removed_nets:
        msg += f" Removed orphan local nets: {removed_nets}."
    return msg


def remove_net(
    G: nx.Graph,
    net_id: str,
    force: bool = False,
    remove_orphan_passives: bool = False,
) -> str:
    """
    删除网络。

    参数：
    - force=False: 若网络上仍挂有 Pin/VirtualPort，则拒绝删除。
    - remove_orphan_passives=True: 删除网络后自动清理“失去所有网络连接”的两端无源器件。
    """
    if not G.has_node(net_id):
        return f"Error: Net '{net_id}' does not exist."
    if G.nodes[net_id].get("type") != "Net":
        return f"Error: Node '{net_id}' is not a Net."

    connected_pins = [
        n
        for n in G.neighbors(net_id)
        if G.nodes[n].get("type") == "Pin"
        and G.get_edge_data(net_id, n).get("relation") == "Connected_To"
    ]
    connected_vports = [
        n
        for n in G.neighbors(net_id)
        if G.nodes[n].get("type") == "VirtualPort"
        and G.get_edge_data(net_id, n).get("relation") == "Connected_To"
    ]

    if (connected_pins or connected_vports) and not force:
        return (
            f"Error: Net '{net_id}' still has {len(connected_pins)} pin connection(s) and "
            f"{len(connected_vports)} virtual-port connection(s). Use force=True to delete."
        )

    touched_components: Set[str] = set()
    for pin in connected_pins:
        for nbr in G.neighbors(pin):
            if (
                G.nodes[nbr].get("type") == "Component"
                and G.get_edge_data(pin, nbr).get("relation") == "Belongs_To"
            ):
                touched_components.add(nbr)

    G.remove_node(net_id)

    removed_orphan_components: List[str] = []
    if remove_orphan_passives:
        passive_types = {
            name.lower()
            for name, cfg in _PASSIVE_TYPE_CONFIG.items()
            if bool(cfg.get("is_passive", True))
        }
        for comp in sorted(touched_components):
            if not G.has_node(comp):
                continue
            if G.nodes[comp].get("type") != "Component":
                continue

            part_type = str(G.nodes[comp].get("part_type", "")).strip().lower()
            if part_type not in passive_types:
                continue

            pins = [
                n
                for n in G.neighbors(comp)
                if G.nodes[n].get("type") == "Pin"
                and G.get_edge_data(comp, n).get("relation") == "Belongs_To"
            ]
            if not pins:
                continue

            has_any_net = False
            for pin in pins:
                for nbr in G.neighbors(pin):
                    if (
                        G.nodes[nbr].get("type") == "Net"
                        and G.get_edge_data(pin, nbr).get("relation") == "Connected_To"
                    ):
                        has_any_net = True
                        break
                if has_any_net:
                    break

            if has_any_net:
                continue

            for pin in pins:
                if G.has_node(pin):
                    G.remove_node(pin)
            if G.has_node(comp):
                G.remove_node(comp)
            removed_orphan_components.append(comp)

    msg = f"Success: Net '{net_id}' removed."
    if removed_orphan_components:
        msg += f" Removed orphan passive components: {removed_orphan_components}."
    return msg


def add_component(
    G: nx.Graph,
    module_id: str,
    refdes: str,
    part_type: str,
    part_number: str,
    value: str = "",
) -> str:
    """
    向指定模块动态添加器件，并自动实例化标准引脚。

    约束：
    - refdes 必须是编号（如 C1/R1/L1/D1/Y1/F1/Q1）；
    - part_number 必须单独传入，并符合命名规范（如 CAP_10uF、RES_10K、CRYSTAL_8MHz、BJT_NPN_2N3904、MOSFET_N_2N7002）。
    """
    if not G.has_node(module_id):
        return f"Error: Module '{module_id}' does not exist."

    if G.has_node(refdes):
        return f"Error: Component reference '{refdes}' already exists in the design."

    canonical_part_type = _canonicalize_passive_type(part_type)
    supported_types = sorted(_PASSIVE_TYPE_CONFIG.keys())
    if canonical_part_type is None:
        return (
            f"Error: Dynamic creation of complex part type '{part_type}' is not allowed here. "
            f"Supported dynamic types are: {supported_types}."
        )

    refdes_err = _validate_passive_refdes(refdes, canonical_part_type)
    if refdes_err:
        return refdes_err

    part_number_err = _validate_passive_part_number(part_number, canonical_part_type, value)
    if part_number_err:
        return part_number_err

    G.add_node(
        refdes,
        type="Component",
        refdes=refdes,
        part_number=part_number,
        part_type=canonical_part_type,
        value=value,
        description=f"Dynamically added {canonical_part_type}",
    )
    G.add_edge(refdes, module_id, relation="Belongs_To")

    cfg = _PASSIVE_TYPE_CONFIG[canonical_part_type]
    pin_names = cfg.get("pin_names") or ["1", "2"]

    for idx, pin_name in enumerate(pin_names, start=1):
        pin_id = f"{refdes}_Pin{idx}"
        G.add_node(
            pin_id,
            type="Pin",
            name=str(pin_name),
            pin_number=str(idx),
            electrical_type="Passive",
        )
        G.add_edge(pin_id, refdes, relation="Belongs_To")

    return (
        f"Success: {canonical_part_type} '{refdes}' ({part_number}, {value}) and its {len(pin_names)} pins added to {module_id}."
    )


def add_passive_and_connect(
    G: nx.Graph,
    module_id: str,
    refdes: str,
    part_type: str,
    part_number: str,
    value: str,
    pin_a_net: str,
    pin_b_net: str,
) -> str:
    """
    高频快捷操作：添加两端器件并将两端分别连接到两个网络。
    """
    canonical_part_type = _canonicalize_passive_type(part_type)
    if canonical_part_type is None:
        supported_types = sorted(_PASSIVE_TYPE_CONFIG.keys())
        return (
            f"Error: Unsupported dynamic component type '{part_type}'. "
            f"Supported types are: {supported_types}."
        )

    pin_names = _PASSIVE_TYPE_CONFIG[canonical_part_type].get("pin_names") or ["1", "2"]
    if len(pin_names) != 2:
        return (
            f"Error: add_passive_and_connect only supports 2-pin components. "
            f"Type '{canonical_part_type}' has {len(pin_names)} pins. "
            "Use add_component + connect_nodes for per-pin wiring."
        )

    add_msg = add_component(G, module_id, refdes, part_type, part_number, value)
    if not add_msg.startswith("Success"):
        return add_msg

    pin1 = f"{refdes}_Pin1"
    pin2 = f"{refdes}_Pin2"

    if not G.has_node(pin_a_net) or G.nodes[pin_a_net].get("type") != "Net":
        return f"Error: pin_a_net '{pin_a_net}' does not exist or is not a Net."
    if not G.has_node(pin_b_net) or G.nodes[pin_b_net].get("type") != "Net":
        return f"Error: pin_b_net '{pin_b_net}' does not exist or is not a Net."

    msg1 = connect_nodes(G, pin1, pin_a_net)
    if not msg1.startswith("Success"):
        return f"Error: Failed to connect {pin1} -> {pin_a_net}. {msg1}"

    msg2 = connect_nodes(G, pin2, pin_b_net)
    if not msg2.startswith("Success"):
        return f"Error: Failed to connect {pin2} -> {pin_b_net}. {msg2}"

    return f"Success: Passive '{refdes}' inserted between {pin_a_net} and {pin_b_net}."


def stitch_global_modules(G: nx.Graph) -> str:
    """
    Phase 4：全局模块缝合。

    新增能力：
    1) 缝合前检测虚拟端口电气类型冲突。
    2) 缝合前检测 expected_voltage 冲突。
    """
    all_vports = [n for n, attr in G.nodes(data=True) if attr.get("type") == "VirtualPort"]
    if not all_vports:
        return "Info: No virtual ports found. Stitching skipped."

    vp_subgraph = nx.Graph()
    for u, v, data in G.edges(data=True):
        if G.nodes[u].get("type") == "VirtualPort" and G.nodes[v].get("type") == "VirtualPort":
            if data.get("relation") == "Connected_To":
                vp_subgraph.add_edge(u, v)

    for vp in all_vports:
        vp_subgraph.add_node(vp)

    isolated_vports = sorted([vp for vp in all_vports if vp_subgraph.degree(vp) == 0])
    components = list(nx.connected_components(vp_subgraph))

    stitched_count = 0

    for comp in components:
        # ---------- 冲突检测：类型 ----------
        norm_types = {_normalize_virtual_electrical_type(G.nodes[vp].get("electrical_type")) for vp in comp}
        if "power" in norm_types and "ground" in norm_types:
            return (
                "Error: Stitching conflict detected. Virtual ports in one bus group contain both "
                f"Power and Ground types: {sorted(comp)}"
            )
        if "signal" in norm_types and len(norm_types) > 1:
            return (
                "Error: Stitching conflict detected. Signal and power-domain ports are mixed in one group: "
                f"{sorted(comp)}"
            )

        # ---------- 冲突检测：电压 ----------
        voltages = set()
        for vp in comp:
            v = _parse_voltage_value(G.nodes[vp].get("expected_voltage"))
            if v is not None:
                voltages.add(round(v, 6))
        if len(voltages) > 1:
            return f"Error: Stitching conflict detected. Expected voltages mismatch in group {sorted(comp)}: {sorted(voltages)}"

        base_vp = list(comp)[0]
        base_name = G.nodes[base_vp].get("name", "GLOBAL")
        global_net_name = f"GLOBAL_NET_{base_name}"

        suffix = 0
        while G.has_node(global_net_name):
            suffix += 1
            global_net_name = f"GLOBAL_NET_{base_name}_BUS_{suffix}"

        net_type = "Signal"
        net_class = "Signal"
        if "ground" in norm_types:
            net_type = "Ground"
            net_class = "Ground"
        elif "power" in norm_types:
            net_type = "Power"
            net_class = "Power"

        G.add_node(
            global_net_name,
            type="Net",
            name=global_net_name,
            scope="Global",
            net_type=net_type,
            net_class=net_class,
            semantic_role=f"Stitched global bus from {base_name}",
            source_pin="",
        )

        pins_to_migrate = []
        nodes_to_remove = set()

        for vp in comp:
            nodes_to_remove.add(vp)
            local_nets = [
                n
                for n in G.neighbors(vp)
                if G.nodes[n].get("type") == "Net"
                and G.get_edge_data(vp, n).get("relation") == "Connected_To"
            ]

            for local_net in local_nets:
                if G.nodes[local_net].get("scope") != "Global":
                    nodes_to_remove.add(local_net)
                for neighbor in G.neighbors(local_net):
                    if G.nodes[neighbor].get("type") == "Pin":
                        pins_to_migrate.append(neighbor)

        for pin in set(pins_to_migrate):
            if not G.has_edge(pin, global_net_name):
                G.add_edge(pin, global_net_name, relation="Connected_To")

        G.remove_nodes_from(nodes_to_remove)
        stitched_count += 1
        print(f"Stitched {len(comp)} module ports into {global_net_name}")

    remaining_vports = [n for n, attr in G.nodes(data=True) if attr.get("type") == "VirtualPort"]
    if remaining_vports:
        return (
            "Error: Stitching finished with residual VirtualPort nodes: "
            f"{sorted(remaining_vports)}"
        )

    if isolated_vports:
        return (
            "Error: Dangling Virtual Ports detected! All ports declared in virtual_ports must be mapped in macro_connections. "
            f"The following VirtualPort(s) have no source/target routing: {isolated_vports}."
        )

    return f"Success: Stitched {stitched_count} global bus networks."


def validate_graph_structure(G: nx.Graph) -> List[str]:
    """
    结构不变量检查器。

    检查项：
    1) belongs / connected 关系的合法节点对。
    2) Pin / Component / VirtualPort / Net 的归属约束。
    3) 全局网络不挂靠 Module。

    注：
    - 按用户评论，不在这里强制“Pin 至多连一个 Net”。
    """
    errors: List[str] = []

    valid_belongs = {
        frozenset({"Pin", "Component"}),
        frozenset({"Component", "Module"}),
        frozenset({"VirtualPort", "Module"}),
        frozenset({"Net", "Module"}),
    }
    valid_connects = {
        frozenset({"Pin", "Net"}),
        frozenset({"Net", "VirtualPort"}),
        frozenset({"VirtualPort", "VirtualPort"}),
    }

    for u, v, data in G.edges(data=True):
        relation = data.get("relation")
        u_type = G.nodes[u].get("type")
        v_type = G.nodes[v].get("type")
        pair = frozenset({u_type, v_type})

        if relation == "Belongs_To":
            if pair not in valid_belongs:
                errors.append(f"Invalid belongs edge: ({u_type}) {u} - ({v_type}) {v}.")
        elif relation == "Connected_To":
            if pair not in valid_connects:
                errors.append(f"Invalid connects edge: ({u_type}) {u} - ({v_type}) {v}.")
        else:
            errors.append(f"Edge with unknown relation '{relation}': ({u}, {v}).")

        if u_type == "Pin" and v_type == "Pin":
            errors.append(f"Invalid edge: Pin-Pin direct connection ({u}, {v}).")

        if u_type == "Net" and v_type == "Net" and relation == "Connected_To":
            errors.append(f"Invalid edge: Net-Net direct connection ({u}, {v}).")

        if {u_type, v_type} == {"VirtualPort", "Pin"} and relation == "Connected_To":
            errors.append(f"Invalid edge: VirtualPort cannot connect directly to Pin ({u}, {v}).")

    for node, attr in G.nodes(data=True):
        n_type = attr.get("type")
        if n_type == "Pin":
            parents = [
                nbr
                for nbr in G.neighbors(node)
                if G.nodes[nbr].get("type") == "Component"
                and G.get_edge_data(node, nbr).get("relation") == "Belongs_To"
            ]
            if len(parents) != 1:
                errors.append(f"Invalid Pin ownership: {node} belongs_to {len(parents)} Component(s).")

        if n_type == "Component":
            parents = [
                nbr
                for nbr in G.neighbors(node)
                if G.nodes[nbr].get("type") == "Module"
                and G.get_edge_data(node, nbr).get("relation") == "Belongs_To"
            ]
            if len(parents) != 1:
                errors.append(f"Invalid Component ownership: {node} belongs_to {len(parents)} Module(s).")

        if n_type == "VirtualPort":
            parents = [
                nbr
                for nbr in G.neighbors(node)
                if G.nodes[nbr].get("type") == "Module"
                and G.get_edge_data(node, nbr).get("relation") == "Belongs_To"
            ]
            if len(parents) != 1:
                errors.append(f"Invalid VirtualPort ownership: {node} belongs_to {len(parents)} Module(s).")

        if n_type == "Net":
            parents = [
                nbr
                for nbr in G.neighbors(node)
                if G.nodes[nbr].get("type") == "Module"
                and G.get_edge_data(node, nbr).get("relation") == "Belongs_To"
            ]

            if attr.get("scope") == "Global":
                if len(parents) != 0:
                    errors.append(f"Invalid Global Net ownership: {node} should not belong_to any Module.")
            else:
                if len(parents) != 1:
                    errors.append(f"Invalid Local Net ownership: {node} belongs_to {len(parents)} Module(s).")

    return errors


