import os
import json
import re
import sys
import copy
from datetime import datetime
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Set, Optional, Tuple

from networkx.readwrite import json_graph

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from langgraph.graph import StateGraph, START, END
except ImportError:
    StateGraph = None
    START = None
    END = None

# ==========================================
# 导入自定义模块
# ==========================================
from tools.build_graph import (
    build_initial_skeleton_graph,
    stitch_global_modules,
    validate_graph_structure,
    get_component_data,
)
from tools.graph_action import (
    set_current_graph,
    get_current_graph,
    clear_active_module_scope,
)
from tools.erc_verification import ERCIssue, ERCVerificationEngine
from tools.extract_state import inspect_module_state

# ==========================================
# 配置区域
# ==========================================
LLM_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL_NAME = "qwen3.5-flash"
LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY", "EMPTY")

BASE_DESIGN_KNOWLEDGE_FILE = "prompts/base_design_knowledge.md"
PHASE0_BASE_DESIGN_KNOWLEDGE_FILE = "prompts/phase0_base_design_knowledge.md"
PHASE1_BASE_DESIGN_KNOWLEDGE_FILE = "prompts/phase1_base_design_knowledge.md"
OUR_METHOD_PROMPT_FILE = "prompts/our_method.md"
PHASE1_PROMPT_FILE = "prompts/phase1_prompt.md"
NAMING_RULES_PROMPT_FILE = "prompts/naming_rules_prompt.md"
BENCHMARK_CASES_FILE = "benchmark_cases.json"

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE) if OpenAI else None

# ==========================================
# 运行日志与产物
# ==========================================
RUN_LOG_DIR: Optional[Path] = None
RUN_RESPONSE_COUNTER = 0


def _ensure_run_log_dir() -> Path:
    global RUN_LOG_DIR
    if RUN_LOG_DIR is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_LOG_DIR = Path("logs") / "agent_runs" / run_id
        (RUN_LOG_DIR / "responses").mkdir(parents=True, exist_ok=True)
        (RUN_LOG_DIR / "round_snapshots").mkdir(parents=True, exist_ok=True)
    return RUN_LOG_DIR


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _append_runtime_log(title: str, content: str = "") -> None:
    if RUN_LOG_DIR is None:
        return

    line = f"[{datetime.now().isoformat(timespec='seconds')}] [{title}]"
    if content:
        line += f" {content}"

    log_file = RUN_LOG_DIR / "agent_runtime.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")



def _call_llm(client, model_name: str, messages: list) -> Any:
    return client.chat.completions.create(
        model=model_name,
        messages=messages,
    )


def init_run_logging(task_description: str, extra_meta: Optional[Dict[str, Any]] = None) -> str:
    """
    初始化一次运行的日志目录并写入元信息。
    """
    global RUN_LOG_DIR, RUN_RESPONSE_COUNTER
    RUN_LOG_DIR = None
    RUN_RESPONSE_COUNTER = 0

    run_dir = _ensure_run_log_dir()
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": LLM_MODEL_NAME,
        "task_description": task_description,
    }
    if extra_meta:
        meta.update(extra_meta)
    _write_text(run_dir / "run_meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
    _append_runtime_log("RUN_START", f"log_dir={run_dir}")
    return str(run_dir)


def _serialize_function_calls(function_calls: List[Any]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for fc in function_calls:
        payload.append(
            {
                "name": getattr(fc, "name", ""),
                "arguments": getattr(fc, "arguments", ""),
                "call_id": getattr(fc, "call_id", ""),
            }
        )
    return payload


def save_model_response_artifacts(
    response: Any,
    phase: str,
    module_id: str,
    loop_index: int,
    reasoning_content: str,
    raw_content: str,
    function_calls: List[Any],
) -> None:
    """
    保存模型响应产物：思考、文本输出、函数调用和原始响应。
    """
    global RUN_RESPONSE_COUNTER
    if RUN_LOG_DIR is None:
        return

    RUN_RESPONSE_COUNTER += 1
    stem = f"{RUN_RESPONSE_COUNTER:04d}_{phase}_m-{module_id or 'NA'}_r-{loop_index:03d}"
    response_dir = RUN_LOG_DIR / "responses"

    _write_text(response_dir / f"{stem}_reasoning.txt", reasoning_content)
    _write_text(response_dir / f"{stem}_output.txt", raw_content)
    _write_text(
        response_dir / f"{stem}_function_calls.json",
        json.dumps(_serialize_function_calls(function_calls), ensure_ascii=False, indent=2),
    )

    try:
        if hasattr(response, "model_dump_json"):
            raw_dump = response.model_dump_json(indent=2)
        elif hasattr(response, "model_dump"):
            raw_dump = json.dumps(response.model_dump(), ensure_ascii=False, indent=2)
        else:
            raw_dump = str(response)
    except Exception as e:
        raw_dump = f"<raw_response_dump_failed> {e}\n{str(response)}"

    _write_text(response_dir / f"{stem}_raw.json", raw_dump)
    _append_runtime_log("MODEL_RESPONSE", f"saved={stem}")

# ==========================================
# 终端彩色日志打印系统
# ==========================================
class LogColor:
    RESET = "\033[0m"
    SYSTEM = "\033[1;36m"       # 亮青色 (系统节点流转)
    THINKING = "\033[38;5;245m" # 灰色 (思考过程)
    TOOL_CALL = "\033[1;33m"    # 亮黄色 (工具调用指令)
    TOOL_RES = "\033[1;32m"     # 亮绿色 (工具返回结果)
    OUTPUT = "\033[1;37m"       # 亮白色 (模型最终输出)
    ERROR = "\033[1;31m"        # 亮红色 (错误信息)

def print_log(color: str, title: str, content: str = ""):
    """带颜色的终端打印函数，确保只打印一次"""
    if content:
        print(f"{color}[{title}] {content}{LogColor.RESET}")
    else:
        print(f"{color}[{title}]{LogColor.RESET}")

    _append_runtime_log(title, content)

# ==========================================
# 定义 LangGraph 状态机 (State)
# ==========================================
class AgentState(TypedDict, total=False):
    design_id: str
    task_description: str
    component_list: List[Dict[str, Any]]
    phase1_max_retries: int
    phase0_plan: Dict[str, Any]
    completed_modules: List[str]
    active_module: str
    process_metrics: Dict[str, Any]
    compile_result: Dict[str, Any]
    skidl_code: str

def load_prompt(filepath: str) -> str:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def _format_component_list_for_prompt(component_list: Any) -> str:
    """
    将指定器件列表格式化为 prompt 可读文本。
    追加所有器件的引脚列表（去除同种器件的重复）。
    """
    if not isinstance(component_list, list):
        if component_list is None:
            return "[]"
        return str(component_list)

    result = copy.deepcopy(component_list)
    seen_part_numbers = set()

    import json
    import os
    
    # 尝试加载统一的引脚数据
    db_path = ""
    for path in [
        "benchmark/component_data/Extracted_Components.json",
        "component_data/Extracted_Components.json",
        "../component_data/Extracted_Components.json",
        "Extracted_Components.json",
        "tools/Extracted_Components.json",
        "../Extracted_Components.json"
    ]:
        if os.path.exists(path):
            db_path = path
            break
            
    db_components = {}
    if db_path:
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "part_number" in item:
                            db_components[item["part_number"].strip().lower()] = item
        except Exception:
            pass

    for comp in result:
        if not isinstance(comp, dict):
            continue
        pn = str(comp.get("part_number", "")).strip()
        if not pn:
            continue
        
        lower_pn = pn.lower()
        if lower_pn in seen_part_numbers:
            continue
            
        seen_part_numbers.add(lower_pn)
        if lower_pn in db_components:
            db_comp = db_components[lower_pn]
            pins = db_comp.get("pins", [])
            if pins:
                # 仅注入 number 和 name
                simplified_pins = []
                for p in pins:
                    simplified_pins.append({
                        "number": p.get("number"),
                        "name": p.get("name")
                    })
                comp["pins"] = simplified_pins

    return json.dumps(result, ensure_ascii=False, indent=2)


def load_unified_method_prompt(
    module_id: str = "N/A",
    task_description: str = "",
    component_list: Any = None,
) -> str:
    """
    组合新架构提示词：our_method + phase0/base_design_knowledge。
    """
    base_knowledge = load_prompt(BASE_DESIGN_KNOWLEDGE_FILE)
    naming_rules = load_prompt(NAMING_RULES_PROMPT_FILE)
    method_prompt = load_prompt(OUR_METHOD_PROMPT_FILE)
    component_text = _format_component_list_for_prompt(component_list)

    phase0_knowledge = ""
    if os.path.exists(PHASE0_BASE_DESIGN_KNOWLEDGE_FILE):
        phase0_knowledge = load_prompt(PHASE0_BASE_DESIGN_KNOWLEDGE_FILE)
        if "{naming_rules}" in phase0_knowledge:
            phase0_knowledge = phase0_knowledge.replace("{naming_rules}", naming_rules)

    if "{naming_rules}" in base_knowledge:
        merged_base_knowledge = base_knowledge.replace("{naming_rules}", naming_rules)
    else:
        # 兼容旧版 base prompt：未提供占位符时回退到自动追加。
        merged_base_knowledge = (
            base_knowledge
            + "\n\n---\n\n"
            + "## Naming Rules (Strictly Enforced)\n"
            + naming_rules
        )

    if "{phase0_base_design_knowledge}" in method_prompt:
        merged = method_prompt.replace(
            "{phase0_base_design_knowledge}",
            phase0_knowledge or merged_base_knowledge,
        )
    else:
        merged = method_prompt.replace("{base_design_knowledge}", merged_base_knowledge)
    merged = merged.replace("{module_id}", module_id)
    merged = merged.replace("{task_description}", task_description)
    merged = merged.replace("{component_list}", component_text)
    return merged


def load_benchmark_cases(cases_file: str = BENCHMARK_CASES_FILE) -> List[Dict[str, Any]]:
    """
    读取 benchmark_cases.json。
    """
    if not os.path.exists(cases_file):
        raise FileNotFoundError(f"Benchmark cases file not found: {cases_file}")

    with open(cases_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Invalid benchmark format in {cases_file}: expected list")
    return data


def get_benchmark_case_by_index(case_index: int, cases_file: str = BENCHMARK_CASES_FILE) -> Dict[str, Any]:
    """
    按 1-based 序号获取测试案例。
    """
    cases = load_benchmark_cases(cases_file)
    if case_index < 1 or case_index > len(cases):
        raise IndexError(f"Case index out of range: {case_index}. Valid range: 1..{len(cases)}")
    return cases[case_index - 1]


def build_initial_state_from_case_index(
    case_index: int,
    cases_file: str = BENCHMARK_CASES_FILE,
) -> Dict[str, Any]:
    case = get_benchmark_case_by_index(case_index, cases_file)
    return {
        "design_id": str(case.get("design_id", "")),
        "task_description": str(case.get("task_description", "")).strip(),
        "component_list": case.get("component_list", []),
    }


def _get_parent_component_of_pin(G, pin_id: str) -> str | None:
    """
    从 Pin 节点反查其归属器件。
    """
    if not G.has_node(pin_id):
        return None
    for nbr in G.neighbors(pin_id):
        if (
            G.nodes[nbr].get("type") == "Component"
            and G.get_edge_data(pin_id, nbr).get("relation") == "Belongs_To"
        ):
            return nbr
    return None


def _component_ready_for_erc(G, comp_id: str) -> bool:
    """
    启发式“器件完成度”判断：
    - 用户要求在器件级别完成后再做 ERC，而不是每次接线都触发。
    - 因此这里采用“轮次批处理 + 最低连接阈值”的折中方案。
    """
    if not G.has_node(comp_id):
        return False

    pins = [n for n in G.neighbors(comp_id) if G.nodes[n].get("type") == "Pin"]
    if not pins:
        return False

    connected_pin_count = 0
    for pin in pins:
        has_net = any(G.nodes[nbr].get("type") == "Net" for nbr in G.neighbors(pin))
        if has_net:
            connected_pin_count += 1

    if connected_pin_count == 0:
        return False

    part_type = str(G.nodes[comp_id].get("part_type", "")).strip().lower()
    if part_type in {
        "resistor",
        "capacitor",
        "inductor",
        "diode",
        "led",
        "crystal",
        "resonator",
        "fuse",
        "mosfet_n",
        "mosfet_p",
        "bjt_npn",
        "bjt_pnp",
    }:
        # 两端无源与三端分立器件通常要求全部引脚接好后再检查。
        return connected_pin_count >= len(pins)

    # 有源器件不要求全引脚接完；至少具备基本连线后进行 ERC。
    return connected_pin_count >= min(2, len(pins))


def _get_component_rules(G, comp_id: str) -> List[Dict[str, Any]]:
    """
    从本地器件库中读取指定器件的 erc_rules。
    """
    if not G.has_node(comp_id):
        return []

    part_number = G.nodes[comp_id].get("part_number")
    if not part_number:
        return []

    item = get_component_data(str(part_number))
    if not item:
        return []

    rules = item.get("erc_rules", [])
    if isinstance(rules, list):
        return rules
    return []


def _run_component_level_erc(
    G,
    component_ids: Set[str],
    require_ready: bool = True,
) -> List[ERCIssue]:
    """
    对受影响器件做批量局部 ERC（A/B 类），返回结构化违规列表（含 severity）。
    """
    engine = ERCVerificationEngine(G)
    violations: List[ERCIssue] = []

    for comp_id in sorted(component_ids):
        if require_ready and not _component_ready_for_erc(G, comp_id):
            continue

        violations.extend(engine.verify_pin_multiplexing(comp_id))

        rules = _get_component_rules(G, comp_id)
        if not rules:
            continue

        for rule in rules:
            if not ERCVerificationEngine.is_local_rule(rule):
                continue
            violations.extend(engine.verify_rule(comp_id, rule))

    return violations


def _module_component_ids(G, module_id: str) -> Set[str]:
    """
    收集指定模块下的全部器件节点。
    """
    if not G.has_node(module_id):
        return set()

    result: Set[str] = set()
    for nbr in G.neighbors(module_id):
        if (
            G.nodes[nbr].get("type") == "Component"
            and G.get_edge_data(module_id, nbr).get("relation") == "Belongs_To"
        ):
            result.add(nbr)
    return result


def save_round_snapshot(G, module_id: str, loop_index: int, tag: str = "post_round") -> None:
    """
    保存每轮后的电路展开图：
    - 当前模块全展开 YAML 视图
    - 全图 node-link JSON
    """
    if RUN_LOG_DIR is None:
        return

    safe_tag = re.sub(r"[^a-zA-Z0-9_-]+", "_", tag)
    stem = f"{module_id}_round_{loop_index:03d}_{safe_tag}"

    expanded_components = sorted(_module_component_ids(G, module_id))
    module_yaml = inspect_module_state(
        G,
        module_id,
        expand_components=expanded_components,
        include_nets=True,
        include_ref_circuits=True,
    )

    snapshot_dir = RUN_LOG_DIR / "round_snapshots"
    _write_text(snapshot_dir / f"{stem}.yaml", module_yaml)

    graph_data = json_graph.node_link_data(G, edges="links")
    _write_text(
        snapshot_dir / f"{stem}.json",
        json.dumps(graph_data, ensure_ascii=False, indent=2),
    )

    _append_runtime_log("ROUND_SNAPSHOT", f"saved={stem}")


def _extract_connection_set_raw(G) -> Dict[str, List[Dict[str, Any]]]:
    """
    从最终图提取 Connection Set（Refdes.PinName 格式）。
    """
    components: List[Dict[str, Any]] = []
    nets: List[Dict[str, Any]] = []

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

            connected_pins: List[str] = []
            for pin_node in connected_pin_nodes:
                connected_pins.append(_pin_ref_from_node(G, pin_node))
            connected_pins = sorted(set(connected_pins))

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


def export_final_graph_artifacts(
    G,
    state: AgentState,
    compile_result: Dict[str, Any],
) -> Dict[str, str]:
    """
    导出最终图产物（标准 JSON），供后续 Pass@k 等评估直接使用。
    """
    if RUN_LOG_DIR is None:
        return {}

    graph_snapshot = json_graph.node_link_data(G, edges="links")
    connection_set = _extract_connection_set_raw(G)

    graph_snapshot_path = RUN_LOG_DIR / "final_graph_snapshot.json"
    connection_set_path = RUN_LOG_DIR / "final_connection_set.json"
    final_result_path = RUN_LOG_DIR / "final_result.json"

    _write_text(graph_snapshot_path, json.dumps(graph_snapshot, ensure_ascii=False, indent=2))
    _write_text(connection_set_path, json.dumps(connection_set, ensure_ascii=False, indent=2))

    final_result = {
        "meta": {
            "design_id": str(state.get("design_id", "")),
            "model": LLM_MODEL_NAME,
        },
        "compile_result": compile_result,
        "connection_set": connection_set,
        "graph_snapshot": graph_snapshot,
    }
    _write_text(final_result_path, json.dumps(final_result, ensure_ascii=False, indent=2))

    artifacts = {
        "graph_snapshot_file": str(graph_snapshot_path),
        "connection_set_file": str(connection_set_path),
        "final_result_file": str(final_result_path),
    }
    _append_runtime_log("FINAL_EXPORT", json.dumps(artifacts, ensure_ascii=False))
    return artifacts


def _run_global_level_erc(G) -> List[ERCIssue]:
    """
    在全局缝合完成后执行 C 类 ERC（属性传播型）检查。
    """
    engine = ERCVerificationEngine(G)
    violations: List[ERCIssue] = []

    component_ids = sorted([n for n, attr in G.nodes(data=True) if attr.get("type") == "Component"])
    for comp_id in component_ids:
        rules = _get_component_rules(G, comp_id)
        if not rules:
            continue

        for rule in rules:
            if not ERCVerificationEngine.is_global_rule(rule):
                continue
            violations.extend(engine.verify_rule(comp_id, rule))

    return violations


def _format_erc_issue(issue: ERCIssue) -> str:
    severity = str(issue.get("severity", "warning")).lower()
    sev_tag = "ERROR" if severity == "error" else "WARNING"
    rule_type = issue.get("rule_type", "UNKNOWN_RULE")
    message = issue.get("message", "")
    description = issue.get("description", "")
    return f"[{sev_tag}] {rule_type}: {message} (Rule: {description})"


def _format_global_erc_report(global_hard: List[ERCIssue], global_soft: List[ERCIssue]) -> str:
    lines = ["全局 ERC（C 类）发现硬违规："]
    for issue in global_hard[:30]:
        lines.append(f"- {_format_erc_issue(issue)}")
    if global_soft:
        lines.append("附加软违规提示：")
        for issue in global_soft[:10]:
            lines.append(f"- {_format_erc_issue(issue)}")
    return "\n".join(lines)

# ==========================================
# 解析 Response 的通用函数
# ==========================================
def parse_and_print_response(
    response,
    phase: str,
    module_id: str = "",
    loop_index: int = 0,
) -> tuple[str, str, List[Any]]:
    """
    遍历 response.output 或 choices，进行彩色日志打印，并提取文本和函数调用
    """
    raw_content = ""
    reasoning_content = ""
    function_calls = []

    if hasattr(response, "output"):
        for item in response.output:
            if item.type == "reasoning":
                summary_list = getattr(item, 'summary', [])
                for s in summary_list:
                    reasoning_content += getattr(s, 'text', '') + "\n"
            elif item.type == "message":
                content_list = getattr(item, 'content', [])
                for c in content_list:
                    raw_content += getattr(c, 'text', '')
            elif item.type == "function_call":
                function_calls.append(item)
    elif hasattr(response, "choices"):
        choice = response.choices[0]
        raw_content = choice.message.content or ""
        if hasattr(choice.message, "reasoning_content"):
            reasoning_content = choice.message.reasoning_content or ""
        # Function calls omit for standard chat due to NO tool requirement

    # 打印提取出的核心内容
    if reasoning_content.strip():
        print_log(LogColor.THINKING, "思考过程", f"\n{reasoning_content.strip()}")
        
    if raw_content.strip():
        print_log(LogColor.OUTPUT, "文本输出", f"\n{raw_content.strip()}")

    save_model_response_artifacts(
        response=response,
        phase=phase,
        module_id=module_id,
        loop_index=loop_index,
        reasoning_content=reasoning_content,
        raw_content=raw_content,
        function_calls=function_calls,
    )

    return raw_content, reasoning_content, function_calls

# ==========================================
# 节点 1：统一设计节点（Phase0 规划 + Phase1+ 连线）
# ==========================================
def _extract_phase0_json(raw_content: str) -> Dict[str, Any]:
    """
    从模型文本输出中提取 Phase0 JSON。
    """
    clean_content = re.sub(r'^```json\s*', '', raw_content, flags=re.MULTILINE)
    clean_content = re.sub(r'^```\s*', '', clean_content, flags=re.MULTILINE).strip()

    match = re.search(r'(\{.*\})', clean_content, re.DOTALL)
    if match:
        clean_content = match.group(1)

    try:
        return json.loads(clean_content)
    except Exception as e:
        print_log(LogColor.ERROR, "JSON解析失败", f"{e}\n原始文本: {clean_content}")
        raise e


def _module_id_from_endpoint(endpoint: str) -> str:
    token = str(endpoint or "").strip()
    if "." in token:
        return token.split(".", 1)[0].strip()
    return ""


def _port_name_from_endpoint(endpoint: str) -> str:
    token = str(endpoint or "").strip()
    if "." in token:
        return token.split(".", 1)[1].strip()
    return ""


def _is_ground_like_name(token: str) -> bool:
    text = str(token or "").strip().lower()
    if not text:
        return False
    ground_tokens = ["gnd", "ground", "agnd", "dgnd", "pgnd", "sgnd", "earth", "chassis"]
    return any(t in text for t in ground_tokens)


def _is_power_like_name(token: str) -> bool:
    text = str(token or "").strip().lower()
    if not text:
        return False
    power_tokens = ["vcc", "vdd", "vin", "vout", "vbat", "vref", "vdda", "vddio", "avdd", "dvdd"]
    if any(t in text for t in power_tokens):
        return True
    return bool(re.search(r"\b\d+v\d*\b", text))


def _normalize_net_name_token(token: str) -> str:
    return str(token or "").strip().upper()


def _collect_global_power_ground_net_names(phase0_plan: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    if not isinstance(phase0_plan, dict):
        return out

    raw_global = phase0_plan.get("initial_global_nets", [])
    if not isinstance(raw_global, list):
        return out

    for net in raw_global:
        if not isinstance(net, dict):
            continue
        name = str(net.get("net_name", net.get("name", ""))).strip()
        if not name:
            continue

        net_type = _normalize_net_type(str(net.get("net_type", "Signal")))
        if net_type in {"Ground", "Power"} or _is_ground_like_name(name) or _is_power_like_name(name):
            out.add(_normalize_net_name_token(name))

    return out


def _normalize_phase0_plan(
    phase0_plan: Dict[str, Any],
    component_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    归一化 Phase 0 输出字段，兼容新旧 schema。

    目标字段：
    - modules[].components: ["U1", "U2"]
    - macro_connections[].source / target / net_name
    """
    if not isinstance(phase0_plan, dict):
        return {
            "initial_global_nets": [],
            "modules": [],
            "macro_connections": [],
        }

    normalized = dict(phase0_plan)

    # 1) initial_global_nets
    raw_global_nets = normalized.get("initial_global_nets", [])
    if not isinstance(raw_global_nets, list):
        raw_global_nets = []

    global_nets: List[Dict[str, Any]] = []
    for idx, net in enumerate(raw_global_nets):
        if not isinstance(net, dict):
            print_log(LogColor.ERROR, "Phase0 Warning", f"initial_global_nets[{idx}] 不是对象，已跳过。")
            continue

        net_name = str(net.get("net_name", net.get("name", ""))).strip()
        if not net_name:
            print_log(LogColor.ERROR, "Phase0 Warning", f"initial_global_nets[{idx}] 缺少 net_name，已跳过。")
            continue

        global_nets.append(
            {
                "net_name": net_name,
                "net_type": str(net.get("net_type", "Power")),
                "semantic_role": net.get("semantic_role", ""),
            }
        )

    for net in global_nets:
        name = str(net.get("net_name", "")).strip()
        net_type = _normalize_net_type(str(net.get("net_type", "")))
        if net_type == "Power" and name.upper().startswith("VCC_"):
            print_log(
                LogColor.ERROR,
                "Phase0 Warning",
                (
                    f"内部生成电源 '{name}' 不应出现在 initial_global_nets；"
                    "请在 provider module 的 virtual_ports/macro_connections 中显式路由。"
                ),
            )

    normalized["initial_global_nets"] = global_nets
    global_power_ground_nets = _collect_global_power_ground_net_names({"initial_global_nets": global_nets})

    # 2) modules[].components
    component_lookup: Dict[str, Dict[str, Any]] = {}
    if isinstance(component_list, list):
        for item in component_list:
            if not isinstance(item, dict):
                continue
            refdes = str(item.get("refdes", "")).strip()
            if refdes:
                component_lookup[refdes] = item

    raw_modules = normalized.get("modules", [])
    if not isinstance(raw_modules, list):
        raw_modules = []

    modules: List[Dict[str, Any]] = []
    for idx, module in enumerate(raw_modules):
        if not isinstance(module, dict):
            print_log(LogColor.ERROR, "Phase0 Warning", f"modules[{idx}] 不是对象，已跳过。")
            continue

        module_id = str(module.get("module_id", "")).strip()
        if not module_id:
            print_log(LogColor.ERROR, "Phase0 Warning", f"modules[{idx}] 缺少 module_id，已跳过。")
            continue

        raw_components = module.get("components", None)
        if not isinstance(raw_components, list):
            # 旧字段兼容：main_components
            legacy = module.get("main_components", [])
            raw_components = []
            if isinstance(legacy, list):
                for c in legacy:
                    if isinstance(c, dict) and str(c.get("refdes", "")).strip():
                        raw_components.append(str(c.get("refdes", "")).strip())

        normalized_components: List[str] = []
        seen_refdes: Set[str] = set()
        for item in raw_components:
            refdes = ""
            if isinstance(item, dict):
                refdes = str(item.get("refdes", "")).strip()
            else:
                refdes = str(item).strip()

            if not refdes or refdes in seen_refdes:
                continue

            if component_lookup and refdes not in component_lookup:
                print_log(
                    LogColor.SYSTEM,
                    "Phase0 Note",
                    f"模块 {module_id} 引用了未在 component_list 中找到的器件 {refdes}。",
                )

            normalized_components.append(refdes)
            seen_refdes.add(refdes)

        module_out = dict(module)
        module_out["components"] = normalized_components
        modules.append(module_out)

    normalized["modules"] = modules

    # 3) macro_connections[].source/target/net_name
    raw_macro = normalized.get("macro_connections", [])
    if not isinstance(raw_macro, list):
        raw_macro = []

    macro_connections: List[Dict[str, Any]] = []
    for idx, conn in enumerate(raw_macro):
        if not isinstance(conn, dict):
            print_log(LogColor.ERROR, "Phase0 Warning", f"macro_connections[{idx}] 不是对象，已跳过。")
            continue

        source = str(conn.get("source", conn.get("source_port", ""))).strip()
        target = str(conn.get("target", conn.get("target_port", ""))).strip()
        if not source or not target:
            print_log(LogColor.ERROR, "Phase0 Warning", f"macro_connections[{idx}] 缺少 source/target，已跳过。")
            continue

        net_name = str(conn.get("net_name", "")).strip()
        if not net_name:
            net_name = _port_name_from_endpoint(source) or _port_name_from_endpoint(target)
        if not net_name:
            net_name = f"AUTO_NET_{idx + 1}"

        src_port = _port_name_from_endpoint(source)
        tgt_port = _port_name_from_endpoint(target)
        if (
            _normalize_net_name_token(net_name) in global_power_ground_nets
            or _normalize_net_name_token(src_port) in global_power_ground_nets
            or _normalize_net_name_token(tgt_port) in global_power_ground_nets
        ):
            print_log(
                LogColor.SYSTEM,
                "Phase0 Note",
                (
                    f"macro_connections[{idx}] 使用全局地网络 '{net_name or src_port or tgt_port}'，"
                    "已移除（该类网络仅通过 initial_global_nets 表达，不参与 macro_connections）。"
                ),
            )
            continue

        macro_connections.append(
            {
                "source": source,
                "target": target,
                "net_name": net_name,
                "description": str(conn.get("description", "")).strip(),
            }
        )

    # 4) 不在 Phase0 阶段按端点强制统一 net_name。
    # 保留 source/target/net_name 原始语义，冲突留到最终合并阶段处理。
    endpoint_to_nets: Dict[Tuple[str, str], Set[str]] = {}
    for conn in macro_connections:
        source = str(conn.get("source", "")).strip()
        target = str(conn.get("target", "")).strip()
        net_name = str(conn.get("net_name", "")).strip()

        src_mod = _module_id_from_endpoint(source)
        src_port = _port_name_from_endpoint(source)
        if src_mod and src_port and net_name:
            endpoint_to_nets.setdefault((src_mod, src_port), set()).add(net_name)

        tgt_mod = _module_id_from_endpoint(target)
        tgt_port = _port_name_from_endpoint(target)
        if tgt_mod and tgt_port and net_name:
            endpoint_to_nets.setdefault((tgt_mod, tgt_port), set()).add(net_name)

    for endpoint, mapped_nets in sorted(endpoint_to_nets.items()):
        if len(mapped_nets) <= 1:
            continue
        # GND/AGND/DGND 这类端点允许保留多网语义（后续可通过单点桥接表达）。
        if all(_is_ground_like_name(n) for n in mapped_nets):
            continue
        print_log(
            LogColor.SYSTEM,
            "Phase0 Note",
            (
                f"端点 {endpoint[0]}.{endpoint[1]} 关联多个网络 {sorted(mapped_nets)}；"
                "已保留原始 macro_connections，并将在最终合并阶段按 net_name 处理。"
            ),
        )

    # 规范 modules[].virtual_ports 结构，并按端点名称自动补齐缺失端口（不改名）。
    for module in modules:
        if not isinstance(module, dict):
            continue

        module_id = str(module.get("module_id", "")).strip()
        if not module_id:
            continue

        raw_vports = module.get("virtual_ports", [])
        vports_out: List[Dict[str, Any]] = []
        if isinstance(raw_vports, list):
            for vp in raw_vports:
                if not isinstance(vp, dict):
                    token = str(vp).strip()
                    if not token:
                        continue
                    vp_obj: Dict[str, Any] = {
                        "port_name": token,
                        "electrical_type": "Signal",
                        "direction": "Bi-dir",
                    }
                else:
                    vp_obj = dict(vp)

                port_name = str(vp_obj.get("port_name", vp_obj.get("name", ""))).strip()
                if not port_name:
                    continue

                if _normalize_net_name_token(port_name) in global_power_ground_nets:
                    print_log(
                        LogColor.SYSTEM,
                        "Phase0 Note",
                        (
                            f"模块 {module_id} 的 virtual_port '{port_name}' 属于全局地网络，"
                            "已从 virtual_ports 中移除。"
                        ),
                    )
                    continue

                vp_obj["port_name"] = port_name
                if not str(vp_obj.get("electrical_type", "")).strip():
                    vp_obj["electrical_type"] = "Signal"
                if not str(vp_obj.get("direction", "")).strip():
                    vp_obj["direction"] = "Bi-dir"
                vports_out.append(vp_obj)

        # 若 macro_connections 端点未出现在 virtual_ports，按端点名称自动补齐。
        expected = [
            port
            for (mid, port) in endpoint_to_nets.keys()
            if (
                mid == module_id
                and str(port).strip()
                and _normalize_net_name_token(port) not in global_power_ground_nets
            )
        ]
        existing_names = {str(v.get("port_name", "")).strip() for v in vports_out if isinstance(v, dict)}
        for port_name in expected:
            if port_name in existing_names:
                continue
            vports_out.append(
                {
                    "port_name": port_name,
                    "electrical_type": "Signal",
                    "direction": "Bi-dir",
                }
            )
            existing_names.add(port_name)
            print_log(
                LogColor.SYSTEM,
                "Phase0 Note",
                f"模块 {module_id} 自动补全 virtual_port '{port_name}'（来源：macro_connections 端点）。",
            )

        # 去重：同名端口保留首个，后续仅补齐空字段。
        dedup: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for vp in vports_out:
            name = str(vp.get("port_name", "")).strip()
            if not name:
                continue
            if name not in dedup:
                dedup[name] = dict(vp)
                order.append(name)
                continue
            for k, v in vp.items():
                if k not in dedup[name] or dedup[name][k] in {"", None}:
                    dedup[name][k] = v

        module["virtual_ports"] = [dedup[name] for name in order]

    # 若同时存在 GND/AGND/DGND，全局规划中必须指定汇聚/分发模块，并在 description 明确单点桥接方案。
    global_name_tokens = {
        str(net.get("net_name", net.get("name", ""))).strip().upper()
        for net in global_nets
        if isinstance(net, dict) and str(net.get("net_name", net.get("name", ""))).strip()
    }
    if {"GND", "AGND", "DGND"}.issubset(global_name_tokens) and modules:
        distribution_module = None
        for module in modules:
            if not isinstance(module, dict):
                continue
            mid = str(module.get("module_id", "")).strip().lower()
            if "power" in mid or "connector" in mid:
                distribution_module = module
                break

        if distribution_module is None:
            distribution_module = modules[0]
            print_log(
                LogColor.ERROR,
                "Phase0 Warning",
                "检测到 GND/AGND/DGND 但未找到明确的 power/connector 汇聚模块，已回退标注到首个模块。",
            )

        note = (
            "Ground distribution plan: AGND and DGND are global domains; "
            "this module is the explicit aggregation/distribution point and must connect AGND and DGND to GND "
            "through a controlled single-point bridge (e.g., net-tie/ferrite/0R per design intent)."
        )
        desc = str(distribution_module.get("description", "")).strip()
        if note not in desc:
            distribution_module["description"] = f"{desc} {note}".strip()
            print_log(
                LogColor.SYSTEM,
                "Phase0 Note",
                f"模块 {distribution_module.get('module_id', '')} 的 description 已补充 GND/AGND/DGND 单点汇聚规划。",
            )

    normalized["macro_connections"] = macro_connections
    normalized["modules"] = modules
    return normalized


def _sort_modules_by_macro_dependencies(phase0_plan: Dict[str, Any]) -> List[str]:
    """
    按 macro_connections 做模块拓扑排序。
    若存在环或异常，则回退到 Phase 0 原始顺序。
    """
    raw_modules = phase0_plan.get("modules", [])
    if not isinstance(raw_modules, list):
        return []

    module_order: List[str] = []
    seen: Set[str] = set()
    for module in raw_modules:
        if not isinstance(module, dict):
            continue
        module_id = str(module.get("module_id", "")).strip()
        if not module_id or module_id in seen:
            continue
        module_order.append(module_id)
        seen.add(module_id)

    if not module_order:
        return []

    index_map = {mid: i for i, mid in enumerate(module_order)}
    outgoing: Dict[str, Set[str]] = {mid: set() for mid in module_order}
    indegree: Dict[str, int] = {mid: 0 for mid in module_order}

    for conn in phase0_plan.get("macro_connections", []):
        if not isinstance(conn, dict):
            continue

        src = _module_id_from_endpoint(conn.get("source", conn.get("source_port", "")))
        tgt = _module_id_from_endpoint(conn.get("target", conn.get("target_port", "")))
        if not src or not tgt or src == tgt:
            continue
        if src not in indegree or tgt not in indegree:
            continue
        if tgt in outgoing[src]:
            continue

        outgoing[src].add(tgt)
        indegree[tgt] += 1

    ready = sorted([mid for mid in module_order if indegree[mid] == 0], key=lambda m: index_map[m])
    sorted_modules: List[str] = []

    while ready:
        current = ready.pop(0)
        sorted_modules.append(current)

        for nxt in sorted(outgoing[current], key=lambda m: index_map[m]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=lambda m: index_map[m])

    if len(sorted_modules) != len(module_order):
        return module_order

    return sorted_modules


def _extract_json_object(raw_content: str) -> Dict[str, Any]:
    """
    从模型输出中提取单个 JSON 对象。
    """
    clean_content = re.sub(r"^```json\s*", "", raw_content, flags=re.MULTILINE)
    clean_content = re.sub(r"^```\s*", "", clean_content, flags=re.MULTILINE).strip()

    match = re.search(r"(\{.*\})", clean_content, re.DOTALL)
    if match:
        clean_content = match.group(1)

    return json.loads(clean_content)


def _component_module_id(G, refdes: str) -> Optional[str]:
    if not G.has_node(refdes):
        return None
    for nbr in G.neighbors(refdes):
        if (
            G.nodes[nbr].get("type") == "Module"
            and G.get_edge_data(refdes, nbr).get("relation") == "Belongs_To"
        ):
            return str(nbr)
    return None


def _component_part_number_from_graph(G, refdes: str) -> str:
    if not G.has_node(refdes):
        return ""
    return str(G.nodes[refdes].get("part_number", "")).strip()


def _pin_nodes_of_component(G, refdes: str) -> List[str]:
    if not G.has_node(refdes):
        return []

    pins: List[str] = []
    for nbr in G.neighbors(refdes):
        if (
            G.nodes[nbr].get("type") == "Pin"
            and G.get_edge_data(refdes, nbr).get("relation") == "Belongs_To"
        ):
            pins.append(str(nbr))
    return pins


def _pin_number_sort_key(value: Any) -> Tuple[int, Any]:
    text = str(value or "").strip()
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def _pin_display_name(pin_attr: Dict[str, Any]) -> str:
    name = str(pin_attr.get("name", "")).strip()
    if name:
        return name
    number = str(pin_attr.get("pin_number", "")).strip()
    if number:
        return number
    return "PIN"


def _pin_ref_from_node(G, pin_node: str) -> str:
    comp = _get_parent_component_of_pin(G, pin_node)
    if not comp:
        return pin_node
    pin_attr = G.nodes[pin_node]
    return f"{comp}.{_pin_display_name(pin_attr)}"


def _net_connected_pin_refs(G, net_name: str) -> List[str]:
    if not G.has_node(net_name):
        return []

    pins = [
        n
        for n in G.neighbors(net_name)
        if (
            G.nodes[n].get("type") == "Pin"
            and G.get_edge_data(net_name, n).get("relation") == "Connected_To"
        )
    ]
    refs = sorted({_pin_ref_from_node(G, p) for p in pins})
    return refs


def _net_connected_pin_refs_for_module(G, net_name: str, module_id: str) -> List[str]:
    """
    返回某网络上“当前模块自身器件”的已连接引脚，避免将其他模块历史连接泄露给当前步骤。
    """
    if not G.has_node(net_name):
        return []

    refs: Set[str] = set()
    for nbr in G.neighbors(net_name):
        if G.nodes[nbr].get("type") != "Pin":
            continue
        if G.get_edge_data(net_name, nbr).get("relation") != "Connected_To":
            continue

        parent_comp = _get_parent_component_of_pin(G, nbr)
        if not parent_comp:
            continue
        if _component_module_id(G, parent_comp) != module_id:
            continue

        refs.add(_pin_ref_from_node(G, nbr))

    return sorted(refs)


def _normalize_net_type(raw_type: str) -> str:
    text = str(raw_type or "Signal").strip().lower()
    if any(x in text for x in ["ground", "gnd", "vss", "-"]):
        return "Ground"
    if any(x in text for x in ["power", "vcc", "vdd", "vin", "vout", "3v3", "5v", "3.3v", "5.0v", "+"]):
        return "Power"
    return "Signal"


def _build_shared_nets_lookup(
    phase0_plan: Dict[str, Any],
    G,
) -> Dict[str, Dict[str, Any]]:
    """
    构建 shared_nets 查找表，优先使用 build_graph 写入的图属性，
    并补充 macro_connections 端点别名网络（source/target 的 port 部分）。
    """
    normalized: Dict[str, Dict[str, Any]] = {}

    graph_shared = G.graph.get("shared_nets")
    if isinstance(graph_shared, dict) and graph_shared:
        for name, meta in graph_shared.items():
            if not isinstance(meta, dict):
                continue
            normalized[str(name)] = {
                "type": _normalize_net_type(str(meta.get("type", "Signal"))),
                "modules": set(meta.get("modules", [])) if isinstance(meta.get("modules", []), list) else set(),
            }

    modules = phase0_plan.get("modules", []) if isinstance(phase0_plan, dict) else []
    module_ids = {
        str(m.get("module_id", "")).strip()
        for m in modules
        if isinstance(m, dict) and str(m.get("module_id", "")).strip()
    }

    # 若图上无 shared_nets，则先根据 phase0_plan 基础字段回退构建。
    for net in phase0_plan.get("initial_global_nets", []):
        if not isinstance(net, dict):
            continue
        name = str(net.get("net_name", net.get("name", ""))).strip()
        if not name:
            continue
        normalized.setdefault(name, {
            "type": _normalize_net_type(str(net.get("net_type", "Power"))),
            "modules": set(module_ids),
        })
        normalized[name]["modules"].update(module_ids)

    def _touch_shared(name: str, net_type_hint: str = "Signal") -> None:
        token = str(name or "").strip()
        if not token:
            return
        if token not in normalized:
            normalized[token] = {
                "type": _normalize_net_type(net_type_hint or token),
                "modules": set(),
            }

    # macro_connections: 除 net_name 外，同时纳入 source/target 的端口别名。
    for conn in phase0_plan.get("macro_connections", []):
        if not isinstance(conn, dict):
            continue

        source = str(conn.get("source", conn.get("source_port", ""))).strip()
        target = str(conn.get("target", conn.get("target_port", ""))).strip()
        src_mod = _module_id_from_endpoint(source)
        tgt_mod = _module_id_from_endpoint(target)

        net_name = str(conn.get("net_name", "")).strip() or _port_name_from_endpoint(source) or _port_name_from_endpoint(target)
        if not net_name:
            continue

        alias_names = {
            net_name,
            _port_name_from_endpoint(source),
            _port_name_from_endpoint(target),
        }

        for alias in alias_names:
            _touch_shared(alias, net_name)
            alias_token = str(alias or "").strip()
            if not alias_token or alias_token not in normalized:
                continue
            if src_mod in module_ids:
                normalized[alias_token]["modules"].add(src_mod)
            if tgt_mod in module_ids:
                normalized[alias_token]["modules"].add(tgt_mod)

    # 补充所有模块自身 Power/Ground 的 virtual_ports 声明为公共网络
    for m in modules:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("module_id", "")).strip()
        if not mid:
            continue

        vports = m.get("virtual_ports", [])
        if isinstance(vports, list):
            for vp in vports:
                if not isinstance(vp, dict):
                    continue
                vp_name = str(vp.get("port_name", vp.get("name", ""))).strip()
                if not vp_name:
                    continue
                vp_type = str(vp.get("type", "Signal")).strip()
                norm_type = _normalize_net_type(vp_type)
                
                # 避免信号端口也被视为全局网络从而导致非短路重名冲突
                if norm_type in {"Power", "Ground"}:
                    _touch_shared(vp_name, norm_type)
                    normalized[vp_name]["modules"].add(mid)

    return normalized


def _derive_module_required_virtual_nets(
    module_id: str,
    module_cfg: Dict[str, Any],
    shared_nets: Dict[str, Dict[str, Any]],
    global_power_ground_nets: Optional[Set[str]] = None,
) -> List[str]:
    """
    从 Phase0 规划推导模块的“必连接口网络”：
    1) module.virtual_ports 中声明的所有端口（不区分信号/电源/地）。
    2) 若未声明 virtual_ports，则回退到 macro/shared_nets 中该模块参与的网络。
    """
    required: Set[str] = set()
    global_power_ground_nets = {
        _normalize_net_name_token(n)
        for n in (global_power_ground_nets or set())
        if str(n).strip()
    }

    raw_vports = module_cfg.get("virtual_ports", []) if isinstance(module_cfg, dict) else []
    has_declared_vports = isinstance(raw_vports, list) and len(raw_vports) > 0

    if isinstance(raw_vports, list):
        for vp in raw_vports:
            if not isinstance(vp, dict):
                continue
            port_name = str(vp.get("port_name", vp.get("name", ""))).strip()
            if not port_name:
                continue
            if _normalize_net_name_token(port_name) in global_power_ground_nets:
                continue
            required.add(port_name)

    # 若模块显式声明了 virtual_ports，则以其为准（最严格且可解释）。
    if required:
        return sorted(required)
    if has_declared_vports:
        return []

    for net_name, meta in shared_nets.items():
        modules = meta.get("modules", set())
        if isinstance(modules, list):
            modules = set(modules)
        if module_id not in modules:
            continue
        if _normalize_net_name_token(str(net_name)) in global_power_ground_nets:
            continue
        required.add(str(net_name))

    return sorted(required)


def _module_has_pin_on_net(G, module_id: str, net_name: str) -> bool:
    if not G.has_node(net_name) or G.nodes[net_name].get("type") != "Net":
        return False

    for nbr in G.neighbors(net_name):
        if G.nodes[nbr].get("type") != "Pin":
            continue
        rel = G.get_edge_data(net_name, nbr).get("relation")
        if rel != "Connected_To":
            continue
        parent_comp = _get_parent_component_of_pin(G, nbr)
        if not parent_comp:
            continue
        if _component_module_id(G, parent_comp) == module_id:
            return True

    return False


def _missing_required_virtual_nets_for_module(
    G,
    module_id: str,
    required_virtual_nets: List[str],
) -> List[str]:
    missing: List[str] = []
    for net_name in required_virtual_nets:
        if not _module_has_pin_on_net(G, module_id, net_name):
            missing.append(net_name)
    return sorted(missing)


def _materialize_module_virtual_nets(
    G,
    module_id: str,
    virtual_net_names: List[str],
    shared_nets: Dict[str, Dict[str, Any]],
) -> None:
    """
    将模块虚拟网络提前写入图（未连线时为空网），保证模型进入模块即可见。
    """
    for net_name in virtual_net_names:
        net_name = str(net_name).strip()
        if not net_name:
            continue

        is_shared = net_name in shared_nets
        if not G.has_node(net_name):
            if is_shared:
                net_type = _normalize_net_type(str(shared_nets.get(net_name, {}).get("type", "Signal")))
                G.add_node(
                    net_name,
                    type="Net",
                    name=net_name,
                    scope="Global",
                    net_type=net_type,
                    net_class=net_type,
                    semantic_role="Phase0 shared net",
                    source_pin="",
                )
            else:
                G.add_node(
                    net_name,
                    type="Net",
                    name=net_name,
                    scope="Local",
                    net_type="Signal",
                    net_class="Signal",
                    semantic_role=f"Phase0 virtual net in {module_id}",
                    source_pin="",
                )
                G.add_edge(net_name, module_id, relation="Belongs_To")
            continue

        if G.nodes[net_name].get("type") != "Net":
            print_log(
                LogColor.ERROR,
                "Phase0 Warning",
                f"虚拟网络 '{net_name}' 与现有非 Net 节点冲突，已跳过预置。",
            )
            continue

        if G.nodes[net_name].get("scope") != "Local":
            continue

        if G.has_edge(net_name, module_id):
            continue

        owner_modules = [
            nbr
            for nbr in G.neighbors(net_name)
            if (
                G.nodes[nbr].get("type") == "Module"
                and G.get_edge_data(net_name, nbr).get("relation") == "Belongs_To"
            )
        ]
        if owner_modules and module_id not in owner_modules:
            print_log(
                LogColor.ERROR,
                "Phase0 Warning",
                (
                    f"虚拟网络 '{net_name}' 已属于模块 {owner_modules}，"
                    f"无法再预置到模块 '{module_id}'。"
                ),
            )
            continue

        G.add_edge(net_name, module_id, relation="Belongs_To")


def _build_module_environment(
    G,
    module_id: str,
    module_cfg: Dict[str, Any],
    phase0_plan: Dict[str, Any],
    shared_nets: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    构建当前模块可见的 Environment Layer。
    """
    global_nets: List[Dict[str, Any]] = []
    for net in phase0_plan.get("initial_global_nets", []):
        if not isinstance(net, dict):
            continue
        name = str(net.get("net_name", net.get("name", ""))).strip()
        if not name:
            continue
        global_nets.append(
            {
                "name": name,
                "type": _normalize_net_type(str(net.get("net_type", "Power"))),
            }
        )

    global_power_ground_nets = _collect_global_power_ground_net_names(phase0_plan)

    virtual_net_names = _derive_module_required_virtual_nets(
        module_id,
        module_cfg,
        shared_nets,
        global_power_ground_nets=global_power_ground_nets,
    )

    virtual_nets: List[Dict[str, Any]] = []
    for net_name in virtual_net_names:
        virtual_nets.append(
            {
                "name": net_name,
                "connected_pins": _net_connected_pin_refs_for_module(G, net_name, module_id),
            }
        )

    module_nets: List[Dict[str, Any]] = []
    for node, attr in G.nodes(data=True):
        if attr.get("type") != "Net":
            continue
        if attr.get("scope") != "Local":
            continue
        if not G.has_edge(node, module_id):
            continue
        if G.get_edge_data(node, module_id).get("relation") != "Belongs_To":
            continue

        module_nets.append(
            {
                "name": str(attr.get("name", node)),
                "pins": _net_connected_pin_refs(G, str(node)),
            }
        )

    passive_naming_context = _build_passive_naming_context(G)

    return {
        "global_nets": sorted(global_nets, key=lambda x: x["name"]),
        "virtual_nets": sorted(virtual_nets, key=lambda x: x["name"]),
        "module_nets": sorted(module_nets, key=lambda x: x["name"]),
        "module_description": str(module_cfg.get("description", "")).strip() if isinstance(module_cfg, dict) else "",
        "existing_passive_count": passive_naming_context["existing_passive_count"],
        "passive_refdes_prefix_hints": passive_naming_context["refdes_prefix_hints"],
    }


def _extract_alt_functions(pin_data: Dict[str, Any]) -> List[str]:
    alt = pin_data.get("alternate_functions")
    if isinstance(alt, list):
        out = [str(x).strip() for x in alt if str(x).strip()]
        if out:
            return out

    desc = str(pin_data.get("description", ""))
    match = re.search(r"alternate functions?\s*:\s*(.+)", desc, flags=re.IGNORECASE)
    if not match:
        return []

    raw = match.group(1)
    candidates = re.split(r"[,;/]", raw)
    return [c.strip() for c in candidates if c.strip()]


def _derive_pin_class(pin_data: Dict[str, Any]) -> str:
    pin_class = str(pin_data.get("pin_class", "")).strip().lower()
    if pin_class in {"power", "system", "gpio"}:
        return pin_class

    et = str(pin_data.get("electrical_type", pin_data.get("type", ""))).strip().lower()
    name = str(pin_data.get("name", "")).strip().upper()

    if "ground" in et or "power" in et:
        return "power"

    if any(tok in name for tok in ["VDD", "VSS", "VDDA", "VSSA", "VBAT", "VCC", "GND", "AGND", "DGND"]):
        return "power"

    if any(tok in name for tok in ["NRST", "RESET", "BOOT", "OSC", "XTAL", "SWD", "JTAG"]):
        return "system"

    return "gpio"


def _match_pin_to_signals(pin_data: Dict[str, Any], needed_signals: Set[str]) -> Optional[str]:
    """
    检查 pin 的 alternate_functions 是否匹配所需信号。
    """
    token_aliases: Dict[str, Set[str]] = {
        "CLK": {"CLK", "SCK", "SCLK", "CK"},
        "SCK": {"SCK", "SCLK", "CLK", "CK"},
        "SCLK": {"SCLK", "SCK", "CLK", "CK"},
        "MOSI": {"MOSI", "SI", "SDI", "SIMO", "COPI"},
        "MISO": {"MISO", "SO", "SDO", "SOMI", "CIPO"},
        "CS": {"CS", "NSS", "SS", "CSN", "NCS"},
        "NCS": {"NCS", "CS", "CSN", "NSS", "SS"},
        "INT": {"INT", "IRQ", "NINT", "INTN"},
        "IRQ": {"IRQ", "INT", "NINT", "INTN"},
        "RST": {"RST", "RESET", "NRST"},
        "RESET": {"RESET", "RST", "NRST"},
    }

    def _part_matches_alt(part: str, alt_upper: str) -> bool:
        variants = token_aliases.get(part, {part})

        # 对短 token 使用词边界匹配，降低误判（例如 "SO" 不应误命中 "SOURCE"）。
        if any(len(v) <= 2 for v in variants):
            for v in variants:
                if re.search(rf"(?<![A-Z0-9]){re.escape(v)}(?![A-Z0-9])", alt_upper):
                    return True
            return False

        return any(v in alt_upper for v in variants)

    alt_funcs = pin_data.get("alternate_functions", [])
    if not isinstance(alt_funcs, list):
        alt_funcs = []
    alt_text = " ".join(str(x) for x in alt_funcs).upper()
    if not alt_text:
        return None

    for signal in sorted(needed_signals):
        parts = [p for p in re.split(r"[_]", signal.upper()) if p]
        if not parts:
            continue

        matched = True
        for part in parts:
            if len(part) <= 1:
                continue
            if re.fullmatch(r"U\d+", part):
                continue

            if part == "UART":
                if "UART" not in alt_text and "USART" not in alt_text:
                    matched = False
                    break
                continue

            if not _part_matches_alt(part, alt_text):
                matched = False
                break

        if matched:
            return signal

    return None


def _normalize_value_token(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "").strip())
    if text:
        return text
    return "GEN"


def _derive_passive_part_number(passive_type: Any, value: Any, fallback: Any = "") -> str:
    explicit = str(fallback or "").strip()
    if explicit:
        return explicit

    ptype = str(passive_type or "").strip().lower()
    token = _normalize_value_token(value)

    type_prefix = {
        "resistor": "RES",
        "capacitor": "CAP",
        "inductor": "IND",
        "diode": "DIODE",
        "led": "LED",
        "crystal": "CRYSTAL",
        "resonator": "CRYSTAL",
        "fuse": "FUSE",
        "mosfet_n": "MOSFET_N",
        "mosfet_p": "MOSFET_P",
        "bjt_npn": "BJT_NPN",
        "bjt_pnp": "BJT_PNP",
    }
    prefix = type_prefix.get(ptype)
    if not prefix:
        prefix = re.sub(r"[^A-Za-z0-9]+", "_", str(passive_type or "PASSIVE").strip().upper())
        prefix = prefix or "PASSIVE"

    return f"{prefix}_{token}"


def _normalize_required_passives(raw_required: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_required, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_required, start=1):
        if not isinstance(item, dict):
            continue

        refdes = str(item.get("refdes", item.get("ref", ""))).strip()
        if not refdes:
            refdes = f"X{idx}"

        passive_type = item.get("type", "")
        value = item.get("value", "")
        part_number = _derive_passive_part_number(passive_type, value, fallback=item.get("part_number", ""))

        normalized.append(
            {
                "refdes": refdes,
                "part_number": part_number,
                "value": str(value).strip(),
                "type": str(passive_type).strip(),
                "role": str(item.get("role", "")).strip(),
            }
        )

    return normalized


def _normalize_topology_nets(raw_topology: Any, current_refdes: str) -> List[Dict[str, Any]]:
    if not isinstance(raw_topology, list):
        return []

    out: List[Dict[str, Any]] = []
    for entry in raw_topology:
        if not isinstance(entry, dict):
            continue

        net_name = str(entry.get("net", entry.get("name", ""))).strip()
        if not net_name:
            continue

        pins: List[str] = []
        raw_pins = entry.get("pins")
        if isinstance(raw_pins, list):
            for pin in raw_pins:
                token = str(pin).strip()
                if token:
                    pins.append(token)
        else:
            nodes = entry.get("nodes", [])
            if isinstance(nodes, list):
                for node in nodes:
                    if not isinstance(node, dict):
                        continue
                    pin_name = str(node.get("pin", node.get("name", ""))).strip()
                    if not pin_name:
                        continue
                    ref = str(node.get("ref", "")).strip()
                    if ref:
                        pins.append(f"{ref}.{pin_name}")
                    else:
                        pins.append(f"{current_refdes}.{pin_name}")

        out.append(
            {
                "net": net_name,
                "pins": pins,
            }
        )

    return out


def _build_component_input_payload(
    G,
    refdes: str,
    environment: Dict[str, Any],
) -> Dict[str, Any]:
    """
    组装逐器件调用的 current_component JSON。
    """
    part_number = _component_part_number_from_graph(G, refdes)
    lib_data = get_component_data(part_number) or {}

    pin_nodes = _pin_nodes_of_component(G, refdes)
    pin_nodes.sort(key=lambda p: _pin_number_sort_key(G.nodes[p].get("pin_number", "")))

    lib_pins = lib_data.get("pins", []) if isinstance(lib_data.get("pins", []), list) else []
    lib_by_num = {str(p.get("number", "")).strip(): p for p in lib_pins if isinstance(p, dict)}
    lib_by_name = {str(p.get("name", "")).strip().upper(): p for p in lib_pins if isinstance(p, dict)}

    pin_infos: List[Dict[str, Any]] = []
    for pin_node in pin_nodes:
        attr = G.nodes[pin_node]
        pin_name = str(attr.get("name", "")).strip()
        pin_number = str(attr.get("pin_number", "")).strip()

        lib_pin = lib_by_num.get(pin_number) or lib_by_name.get(pin_name.upper()) or {}
        merged_pin = {
            "name": pin_name,
            "number": int(pin_number) if pin_number.isdigit() else pin_number,
            "type": str(attr.get("electrical_type", "Passive")),
            "description": str(attr.get("description", "")),
            "pin_class": _derive_pin_class({**lib_pin, **attr}),
            "alternate_functions": _extract_alt_functions({**lib_pin, **attr}),
        }
        pin_infos.append(merged_pin)

    needed_virtual_nets = {
        str(item.get("name", "")).strip()
        for item in environment.get("virtual_nets", [])
        if isinstance(item, dict) and str(item.get("name", "")).strip()
    }

    def _derive_virtual_alt_keywords(virtual_net_names: Set[str]) -> Set[str]:
        """
        从 virtual net 名称提取“应强制保留的 alternate_functions 关键词”。
        命中后对应 GPIO 不应被折叠到 other_available_pins。
        """
        if not virtual_net_names:
            return set()

        family_tokens: Dict[str, Set[str]] = {
            "I2C": {"I2C", "SCL", "SDA", "SMBA"},
            "SPI": {"SPI", "MOSI", "MISO", "SCK", "SCLK", "NSS", "SS", "CS", "CSN"},
            "USB": {"USB", "DP", "DM", "D+", "D-"},
            "UART": {"UART", "USART", "TX", "RX", "RT", "RTS", "CTS"},
            "USART": {"UART", "USART", "TX", "RX", "RT", "RTS", "CTS"},
            "CAN": {"CAN", "TX", "RX"},
            "I2S": {"I2S", "MCK", "SCK", "WS", "SD"},
            "SWD": {"SWD", "SWCLK", "SWDIO"},
            "JTAG": {"JTAG", "JTMS", "JTCK", "JTDI", "JTDO", "NJTRST", "TRACESWO"},
        }
        direct_tokens = {
            "I2C", "SPI", "USB", "UART", "USART", "CAN", "I2S",
            "TX", "RX", "RT", "RTS", "CTS", "SCL", "SDA", "MOSI", "MISO", "SCK", "SCLK",
            "SWD", "JTAG", "DP", "DM",
        }

        out: Set[str] = set()
        for net_name in virtual_net_names:
            up = str(net_name).upper()

            for family, tokens in family_tokens.items():
                if family in up:
                    out.update(tokens)

            parts = [p for p in re.split(r"[^A-Z0-9]+", up) if p]
            for token in parts:
                if token in direct_tokens:
                    out.add(token)

        return out

    virtual_alt_keywords = _derive_virtual_alt_keywords(needed_virtual_nets)

    def _pin_matches_virtual_alt_keywords(pin_obj: Dict[str, Any]) -> bool:
        if not virtual_alt_keywords:
            return False
        alt_funcs = pin_obj.get("alternate_functions", [])
        if not isinstance(alt_funcs, list) or not alt_funcs:
            return False

        alt_text = " ".join(str(x) for x in alt_funcs).upper()
        for token in virtual_alt_keywords:
            if not token:
                continue

            # 对短 token 使用词边界，避免把 "CS" 误命中到 "OSC" 等无关函数名。
            if len(token) <= 2:
                if re.search(rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])", alt_text):
                    return True
                continue

            if token in alt_text:
                return True

        return False

    for pin in pin_infos:
        if pin.get("pin_class") != "gpio":
            continue
        # 仅用于内部筛选“应展开的 GPIO”，不暴露给模型输入。
        pin["has_virtual_match"] = bool(_match_pin_to_signals(pin, needed_virtual_nets))

    reference_circuits = []
    for rc in lib_data.get("reference_circuits", []) if isinstance(lib_data.get("reference_circuits", []), list) else []:
        if not isinstance(rc, dict):
            continue
        topology_nets = _normalize_topology_nets(
            rc.get("topology_nets", rc.get("topology", [])),
            current_refdes=refdes,
        )
        required_passives = _normalize_required_passives(rc.get("required_passives", []))
        reference_circuits.append(
            {
                "name": rc.get("name", rc.get("circuit_name", "default")),
                "topology_nets": topology_nets,
                "required_passives": required_passives,
            }
        )

    erc_rules = []
    for rule in lib_data.get("erc_rules", []) if isinstance(lib_data.get("erc_rules", []), list) else []:
        if not isinstance(rule, dict):
            continue
        params = rule.get("parameters", {}) if isinstance(rule.get("parameters", {}), dict) else {}

        target = ""
        target_pins = params.get("target_pins")
        if isinstance(target_pins, list) and target_pins:
            target = f"{refdes}.{target_pins[0]}"
        elif isinstance(target_pins, str) and target_pins.strip():
            target = f"{refdes}.{target_pins.strip()}"
        elif isinstance(params.get("pin_in"), str):
            target = f"{refdes}.{params.get('pin_in')}"

        erc_rules.append(
            {
                "rule": str(rule.get("rule_type", "UNKNOWN")),
                "target": target,
                "severity": str(rule.get("severity", "warning")),
                "description": str(rule.get("description", "")),
                "parameters": params,
            }
        )

    erc_required_pin_tokens: Set[str] = set()
    for rule in erc_rules:
        target = str(rule.get("target", "")).strip()
        if not target:
            continue

        pin_token = target.split(".", 1)[1] if "." in target else target
        pin_token = pin_token.strip().upper()
        if not pin_token:
            continue

        erc_required_pin_tokens.add(pin_token)
        if pin_token.startswith("PIN") and len(pin_token) > 3:
            erc_required_pin_tokens.add(pin_token[3:])
        if pin_token.isdigit():
            erc_required_pin_tokens.add(f"PIN{pin_token}")

    def _is_erc_required_gpio_pin(pin_obj: Dict[str, Any]) -> bool:
        name_token = str(pin_obj.get("name", "")).strip().upper()
        num_token = str(pin_obj.get("number", "")).strip().upper()
        probe_tokens = {name_token, num_token}
        if num_token:
            probe_tokens.add(f"PIN{num_token}")
        return bool(probe_tokens & erc_required_pin_tokens)

    power_system_pins = [p for p in pin_infos if p.get("pin_class") in {"power", "system"}]
    gpio_pins = [p for p in pin_infos if p.get("pin_class") == "gpio"]
    expanded_gpio_pins = [
        p
        for p in gpio_pins
        if p.get("has_virtual_match") or _is_erc_required_gpio_pin(p) or _pin_matches_virtual_alt_keywords(p)
    ]
    expanded_gpio_keys = {
        (
            str(p.get("name", "")).strip().upper(),
            str(p.get("number", "")).strip().upper(),
        )
        for p in expanded_gpio_pins
    }
    unused_gpio_pins = [
        p
        for p in gpio_pins
        if (
            str(p.get("name", "")).strip().upper(),
            str(p.get("number", "")).strip().upper(),
        )
        not in expanded_gpio_keys
    ]

    def _group_folded_pins_by_class(pins: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        将折叠引脚按 pin_class 分组，仅保留 number/name 两个字段。
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for pin in pins:
            pin_class = str(pin.get("pin_class", "")).strip().lower() or "other"
            grouped.setdefault(pin_class, []).append(
                {
                    "number": pin.get("number", ""),
                    "name": pin.get("name", ""),
                }
            )

        for cls, cls_pins in grouped.items():
            cls_pins.sort(key=lambda x: _pin_number_sort_key(x.get("number", "")))
            grouped[cls] = cls_pins

        return {cls: grouped[cls] for cls in sorted(grouped.keys())}

    payload: Dict[str, Any] = {
        "refdes": refdes,
        "part_number": part_number,
        "reference_circuits": reference_circuits,
        "erc_rules": erc_rules,
    }

    should_fold = len(gpio_pins) > 20
    if should_fold:
        payload["power_and_system_pins"] = [
            {
                "name": p["name"],
                "number": p["number"],
                "type": p["type"],
                "description": p["description"],
            }
            for p in power_system_pins
        ]
        payload["key_gpio_pins"] = [
            {
                "name": p["name"],
                "number": p["number"],
                "type": p["type"],
                "erc_required": _is_erc_required_gpio_pin(p),
                "alt_functions": p.get("alternate_functions", []),
            }
            for p in expanded_gpio_pins
        ]
        payload["other_available_pins"] = _group_folded_pins_by_class(unused_gpio_pins)
    else:
        payload["pins"] = [
            {
                "name": p["name"],
                "number": p["number"],
                "type": p["type"],
                "description": p["description"],
            }
            for p in pin_infos
        ]

    return payload


def _build_phase1_component_prompt() -> str:
    base_knowledge = load_prompt(BASE_DESIGN_KNOWLEDGE_FILE)
    naming_rules = load_prompt(NAMING_RULES_PROMPT_FILE)
    phase1_template = load_prompt(PHASE1_PROMPT_FILE)
    phase1_knowledge = ""

    if os.path.exists(PHASE1_BASE_DESIGN_KNOWLEDGE_FILE):
        phase1_knowledge = load_prompt(PHASE1_BASE_DESIGN_KNOWLEDGE_FILE)
        if "{naming_rules}" in phase1_knowledge:
            phase1_knowledge = phase1_knowledge.replace("{naming_rules}", naming_rules)
        else:
            phase1_knowledge = (
                phase1_knowledge
                + "\n\n---\n\n"
                + "## Naming Rules (Strictly Enforced)\n"
                + naming_rules
            )

    if "{naming_rules}" in base_knowledge:
        merged_base_knowledge = base_knowledge.replace("{naming_rules}", naming_rules)
    else:
        merged_base_knowledge = (
            base_knowledge
            + "\n\n---\n\n"
            + "## Naming Rules (Strictly Enforced)\n"
            + naming_rules
        )

    selected_knowledge = phase1_knowledge or merged_base_knowledge

    if "{phase1_base_design_knowledge}" in phase1_template:
        return phase1_template.replace("{phase1_base_design_knowledge}", selected_knowledge)

    if "{base_design_knowledge}" in phase1_template:
        return phase1_template.replace("{base_design_knowledge}", selected_knowledge)

    return f"{phase1_template}\n\n{selected_knowledge}"


def _normalize_connection_output(
    output: Dict[str, Any],
    current_refdes: str,
    current_part_number: str,
) -> Dict[str, Any]:
    """
    对模型输出做最小规范化，确保后续合并稳定。
    """
    components: List[Dict[str, Any]] = []
    seen_refdes: Set[str] = set()

    raw_components = output.get("components", []) if isinstance(output, dict) else []
    if isinstance(raw_components, list):
        for comp in raw_components:
            if not isinstance(comp, dict):
                continue
            refdes = str(comp.get("refdes", "")).strip()
            if not refdes or refdes in seen_refdes:
                continue
            part_number = str(comp.get("part_number", "")).strip()
            components.append(
                {
                    "refdes": refdes,
                    "part_number": part_number,
                }
            )
            seen_refdes.add(refdes)

    if current_refdes not in seen_refdes:
        components.insert(
            0,
            {
                "refdes": current_refdes,
                "part_number": current_part_number,
            },
        )
        seen_refdes.add(current_refdes)

    def _coerce_pin_refs(raw_pins: Any) -> List[str]:
        items: List[Any] = []

        if isinstance(raw_pins, list):
            items = raw_pins
        elif isinstance(raw_pins, str):
            text = raw_pins.strip()
            if not text:
                return []

            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        items = parsed
                    else:
                        items = [text]
                except Exception:
                    inner = text[1:-1].strip()
                    if inner:
                        items = [p.strip() for p in inner.split(",")]
            else:
                if "," in text:
                    items = [p.strip() for p in text.split(",")]
                else:
                    items = [text]
        else:
            return []

        out: List[str] = []
        for pin_ref in items:
            token = str(pin_ref).strip().strip('"').strip("'")
            if token:
                out.append(token)
        return out

    nets: List[Dict[str, Any]] = []
    raw_nets = output.get("nets", []) if isinstance(output, dict) else []
    if isinstance(raw_nets, list):
        for net in raw_nets:
            if not isinstance(net, dict):
                continue
            name = str(net.get("name", "")).strip()
            if not name:
                continue

            pins = _coerce_pin_refs(net.get("pins", []))

            nets.append(
                {
                    "name": name,
                    "pins": pins,
                }
            )

    return {
        "components": components,
        "nets": nets,
    }


def _dynamic_pin_defs_from_part_number(part_number: str) -> List[Dict[str, str]]:
    token = str(part_number or "").strip().upper()

    if token.startswith(("RES_", "CAP_", "IND_", "CRYSTAL_", "FUSE_", "NTC_", "PTC_")):
        return [
            {"number": "1", "name": "1", "electrical_type": "Passive"},
            {"number": "2", "name": "2", "electrical_type": "Passive"},
        ]

    if token.startswith(("DIODE_", "LED_", "TVS_", "ESD_")):
        return [
            {"number": "1", "name": "A", "electrical_type": "Passive"},
            {"number": "2", "name": "K", "electrical_type": "Passive"},
        ]

    if token.startswith(("MOSFET_N_", "MOSFET_P_")):
        return [
            {"number": "1", "name": "G", "electrical_type": "Passive"},
            {"number": "2", "name": "D", "electrical_type": "Passive"},
            {"number": "3", "name": "S", "electrical_type": "Passive"},
        ]

    if token.startswith(("BJT_NPN_", "BJT_PNP_")):
        return [
            {"number": "1", "name": "B", "electrical_type": "Passive"},
            {"number": "2", "name": "C", "electrical_type": "Passive"},
            {"number": "3", "name": "E", "electrical_type": "Passive"},
        ]

    # 未知动态器件回退为两端口，避免完全无法合并。
    return [
        {"number": "1", "name": "1", "electrical_type": "Passive"},
        {"number": "2", "name": "2", "electrical_type": "Passive"},
    ]


def _is_allowed_added_part_number(part_number: str) -> bool:
    token = str(part_number or "").strip().upper()
    if not token:
        return False

    allowed_prefixes = (
        "RES_",
        "CAP_",
        "IND_",
        "DIODE_",
        "LED_",
        "CRYSTAL_",
        "FUSE_",
        "MOSFET_",
        "BJT_",
        "TVS_",
        "ESD_",
        "NTC_",
        "PTC_",
    )
    return token.startswith(allowed_prefixes)


def _is_passive_component_node(G, component_id: str) -> bool:
    if not G.has_node(component_id):
        return False
    if G.nodes[component_id].get("type") != "Component":
        return False

    comp_attr = G.nodes[component_id]
    part_type = str(comp_attr.get("part_type", "")).strip().lower()
    if part_type in {
        "resistor",
        "capacitor",
        "inductor",
        "diode",
        "led",
        "crystal",
        "resonator",
        "fuse",
        "mosfet_n",
        "mosfet_p",
        "bjt_npn",
        "bjt_pnp",
    }:
        return True

    part_number = str(comp_attr.get("part_number", "")).strip()
    if _is_allowed_added_part_number(part_number):
        return True

    pin_nodes = _pin_nodes_of_component(G, component_id)
    if not pin_nodes or len(pin_nodes) > 3:
        return False

    for pin_node in pin_nodes:
        et = str(G.nodes[pin_node].get("electrical_type", "")).strip().lower()
        if et != "passive":
            return False

    return True


def _count_existing_passive_components(G) -> int:
    count = 0
    for node, attr in G.nodes(data=True):
        if attr.get("type") != "Component":
            continue
        if _is_passive_component_node(G, node):
            count += 1
    return count


def _build_passive_naming_context(G) -> Dict[str, Any]:
    by_prefix: Dict[str, Dict[str, int]] = {}
    total = 0

    for node, attr in G.nodes(data=True):
        if attr.get("type") != "Component":
            continue
        if not _is_passive_component_node(G, str(node)):
            continue

        total += 1
        refdes = str(node).strip().upper()
        match = re.match(r"^([A-Z]+)(\d+)$", refdes)
        if not match:
            continue

        prefix = match.group(1)
        index = int(match.group(2))

        entry = by_prefix.setdefault(prefix, {"count": 0, "max_index": 0})
        entry["count"] += 1
        if index > entry["max_index"]:
            entry["max_index"] = index

    prefix_hints: Dict[str, Dict[str, int]] = {}
    for prefix, entry in sorted(by_prefix.items()):
        max_index = int(entry.get("max_index", 0))
        prefix_hints[prefix] = {
            "count": int(entry.get("count", 0)),
            "next_index": max_index + 1,
        }

    return {
        "existing_passive_count": int(total),
        "refdes_prefix_hints": prefix_hints,
    }


def _instantiate_component_node(
    G,
    module_id: str,
    refdes: str,
    part_number: str,
) -> Tuple[bool, str]:
    if not G.has_node(module_id):
        return False, f"Module '{module_id}' not found"

    if G.has_node(refdes):
        if G.nodes[refdes].get("type") != "Component":
            return False, f"Node '{refdes}' exists and is not a Component"
        if not G.has_edge(refdes, module_id):
            G.add_edge(refdes, module_id, relation="Belongs_To")
        return True, "already exists"

    G.add_node(
        refdes,
        type="Component",
        refdes=refdes,
        part_number=part_number,
        description="Added from component output",
    )
    G.add_edge(refdes, module_id, relation="Belongs_To")

    pin_defs: List[Dict[str, Any]] = []
    lib_data = get_component_data(part_number)
    if lib_data and isinstance(lib_data.get("pins", []), list) and lib_data.get("pins", []):
        for p in lib_data.get("pins", []):
            if not isinstance(p, dict):
                continue
            pin_defs.append(
                {
                    "number": str(p.get("number", "")).strip(),
                    "name": str(p.get("name", "")).strip(),
                    "electrical_type": str(p.get("electrical_type", p.get("type", "Passive"))).strip() or "Passive",
                    "description": str(p.get("description", "")),
                }
            )
    else:
        pin_defs = _dynamic_pin_defs_from_part_number(part_number)

    for pin in pin_defs:
        pin_number = str(pin.get("number", "")).strip()
        if not pin_number:
            continue
        pin_id = f"{refdes}_Pin{pin_number}"
        if G.has_node(pin_id):
            continue
        G.add_node(
            pin_id,
            type="Pin",
            name=str(pin.get("name", pin_number)).strip() or pin_number,
            pin_number=pin_number,
            electrical_type=str(pin.get("electrical_type", "Passive")),
            description=str(pin.get("description", "")),
        )
        G.add_edge(pin_id, refdes, relation="Belongs_To")

    return True, "created"


def _resolve_pin_node_from_ref(G, pin_ref: str) -> Optional[str]:
    token = str(pin_ref or "").strip()
    if not token:
        return None

    if G.has_node(token) and G.nodes[token].get("type") == "Pin":
        return token

    if "." not in token:
        return None

    refdes, pin_token = token.split(".", 1)
    refdes = refdes.strip()
    pin_token = pin_token.strip()
    if not refdes or not pin_token:
        return None
    if not G.has_node(refdes) or G.nodes[refdes].get("type") != "Component":
        return None

    pin_token_upper = pin_token.upper()
    for pin_node in _pin_nodes_of_component(G, refdes):
        attr = G.nodes[pin_node]
        pin_name = str(attr.get("name", "")).strip()
        pin_number = str(attr.get("pin_number", "")).strip()

        if pin_token == pin_name or pin_token == pin_number:
            return pin_node

        if pin_token_upper == pin_name.upper() or pin_token_upper == pin_number.upper():
            return pin_node

        if pin_token_upper == f"PIN{pin_number}".upper():
            return pin_node

    return None


def _merge_component_output_to_graph(
    G,
    module_id: str,
    current_refdes: str,
    output: Dict[str, Any],
    shared_nets: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """
    将单器件 Connection Set 合并到图。
    """
    errors: List[str] = []
    new_nets: List[str] = []

    components = output.get("components", []) if isinstance(output, dict) else []
    nets = output.get("nets", []) if isinstance(output, dict) else []

    allowed_refdes: Set[str] = {current_refdes}

    if isinstance(components, list):
        for comp in components:
            if not isinstance(comp, dict):
                continue

            refdes = str(comp.get("refdes", "")).strip()
            if not refdes:
                continue
            if refdes == current_refdes:
                continue

            part_number = str(comp.get("part_number", "")).strip()
            if not part_number:
                errors.append(f"Added component '{refdes}' missing part_number")
                continue

            if not _is_allowed_added_part_number(part_number):
                errors.append(
                    f"Added component '{refdes}' has unsupported part_number '{part_number}'"
                )
                continue

            if G.has_node(refdes):
                errors.append(
                    f"Component '{refdes}' already exists in design; output must only include current IC and newly added components"
                )
                continue
            else:
                ok, msg = _instantiate_component_node(G, module_id, refdes, part_number)
                if not ok:
                    errors.append(f"Failed to add component '{refdes}': {msg}")
                    continue

            allowed_refdes.add(refdes)

    if isinstance(nets, list):
        def _coerce_pin_refs(raw_pins: Any) -> List[str]:
            if isinstance(raw_pins, list):
                return [str(p).strip() for p in raw_pins if str(p).strip()]

            if isinstance(raw_pins, str):
                text = raw_pins.strip()
                if not text:
                    return []
                if text.startswith("[") and text.endswith("]"):
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list):
                            return [str(p).strip() for p in parsed if str(p).strip()]
                    except Exception:
                        inner = text[1:-1].strip()
                        if inner:
                            return [p.strip().strip('"').strip("'") for p in inner.split(",") if p.strip()]
                if "," in text:
                    return [p.strip().strip('"').strip("'") for p in text.split(",") if p.strip()]
                return [text]

            return []

        for net in nets:
            if not isinstance(net, dict):
                continue

            net_name = str(net.get("name", "")).strip()
            if not net_name:
                errors.append("Net entry missing name")
                continue

            is_shared = net_name in shared_nets
            if not G.has_node(net_name):
                if is_shared:
                    net_type = _normalize_net_type(str(shared_nets.get(net_name, {}).get("type", "Signal")))
                    G.add_node(
                        net_name,
                        type="Net",
                        name=net_name,
                        scope="Global",
                        net_type=net_type,
                        net_class=net_type,
                        semantic_role="Phase0 shared net",
                        source_pin="",
                    )
                else:
                    G.add_node(
                        net_name,
                        type="Net",
                        name=net_name,
                        scope="Local",
                        net_type="Signal",
                        net_class="Signal",
                        semantic_role=f"Module net in {module_id}",
                        source_pin="",
                    )
                    G.add_edge(net_name, module_id, relation="Belongs_To")
                new_nets.append(net_name)
            else:
                if G.nodes[net_name].get("type") != "Net":
                    errors.append(f"Node '{net_name}' exists and is not Net")
                    continue
                if G.nodes[net_name].get("scope") == "Local" and not G.has_edge(net_name, module_id):
                    owner_modules = [
                        nbr
                        for nbr in G.neighbors(net_name)
                        if (
                            G.nodes[nbr].get("type") == "Module"
                            and G.get_edge_data(net_name, nbr).get("relation") == "Belongs_To"
                        )
                    ]
                    if owner_modules and module_id not in owner_modules and not is_shared:
                        errors.append(
                            f"Local net '{net_name}' already belongs to module(s) {owner_modules}, cannot reuse in '{module_id}'"
                        )
                        continue
                    G.add_edge(net_name, module_id, relation="Belongs_To")

            raw_pin_refs = net.get("pins", [])
            pin_refs = _coerce_pin_refs(raw_pin_refs)
            if not isinstance(raw_pin_refs, (list, str)):
                errors.append(f"Net '{net_name}' pins must be a list or string")
                continue

            for pin_ref in pin_refs:
                pin_token = str(pin_ref).strip()
                if not pin_token:
                    continue

                refdes = pin_token.split(".", 1)[0].strip() if "." in pin_token else ""
                if refdes and refdes not in allowed_refdes:
                    errors.append(
                        f"Pin '{pin_token}' references component outside current output scope"
                    )
                    continue

                pin_node = _resolve_pin_node_from_ref(G, pin_token)
                if pin_node is None:
                    errors.append(f"Pin '{pin_token}' not found")
                    continue

                if not G.has_edge(pin_node, net_name):
                    G.add_edge(pin_node, net_name, relation="Connected_To")

    return new_nets, errors


def _build_component_retry_feedback(
    current_refdes: str,
    merge_errors: List[str],
    hard_erc_issues: List[ERCIssue],
    soft_erc_issues: Optional[List[ERCIssue]] = None,
    best_effort_soft: bool = False,
) -> List[Dict[str, Any]]:
    feedback: List[Dict[str, Any]] = []
    critical_repair_rule = (
        "CRITICAL REPAIR RULE: Rebuild one integrated and correct JSON based on erc_errors and your_previous_output. Include all previously-correct components and nets within current output scope plus your fixes. Do NOT output only deltas, and do NOT include historical pins from other modules."
    )

    for err in merge_errors:
        feedback.append(
            {
                "rule": "MERGE_VALIDATION",
                "target": current_refdes,
                "severity": "error",
                "message": err,
            }
        )

    for issue in hard_erc_issues:
        feedback.append(
            {
                "rule": str(issue.get("rule_type", "ERC_RULE")),
                "target": str(issue.get("component_id", current_refdes)),
                "severity": str(issue.get("severity", "error")),
                "message": str(issue.get("message", "")),
            }
        )

    if hard_erc_issues or merge_errors:
        feedback.append(
            {
                "rule": "CRITICAL_REPAIR_OUTPUT_REQUIREMENT",
                "target": current_refdes,
                "severity": "error",
                "message": critical_repair_rule,
            }
        )

    soft_issue_list = soft_erc_issues or []
    if best_effort_soft and soft_issue_list:
        feedback.append(
            {
                "rule": "SOFT_ERC_BEST_EFFORT",
                "target": current_refdes,
                "severity": "warning",
                "message": "Soft ERC warnings still exist. Please revise to reduce them as much as possible while preserving hard-ERC correctness.",
            }
        )

    for issue in soft_issue_list:
        feedback.append(
            {
                "rule": str(issue.get("rule_type", "ERC_RULE")),
                "target": str(issue.get("component_id", current_refdes)),
                "severity": "warning",
                "message": str(issue.get("message", "")),
            }
        )

    return feedback


def design_node(state: AgentState) -> AgentState:
    if client is None:
        raise RuntimeError(
            "openai package is not installed. Please install it (e.g., pip install openai) before running agent.py."
        )

    task_desc = str(state.get("task_description", "")).strip()
    if not task_desc:
        raise ValueError("Missing task_description in state.")

    component_list = state.get("component_list", [])
    design_id = str(state.get("design_id", "")).strip()

    log_dir = init_run_logging(
        task_desc,
        extra_meta={
            "design_id": design_id,
            "component_list": component_list,
        },
    )
    print_log(LogColor.SYSTEM, "Design Node", "启动：按统一方法执行 Phase0 规划与 Phase1+ 连线...")
    print_log(LogColor.SYSTEM, "Run Log", f"日志目录：{log_dir}")

    phase0_system_prompt = load_unified_method_prompt(
        module_id="N/A",
        task_description=task_desc,
        component_list=component_list,
    )
    phase0_user_prompt = (
        "请先执行 Phase 0（宏观规划），只输出严格 JSON 对象，不要输出解释性文字。"
        "必须包含 initial_global_nets、modules、macro_connections；"
        "macro_connections 使用 source/target/net_name 字段。"
    )

    phase0_resp = _call_llm(
        client,
        LLM_MODEL_NAME,
        [
            {"role": "system", "content": phase0_system_prompt},
            {"role": "user", "content": phase0_user_prompt},
        ]
    )

    raw_content, _, _ = parse_and_print_response(
        phase0_resp,
        phase="phase0",
        module_id="N/A",
        loop_index=0,
    )
    phase0_plan = _extract_phase0_json(raw_content)
    phase0_plan = _normalize_phase0_plan(phase0_plan, component_list)

    planned_global_nets = phase0_plan.get("initial_global_nets", [])
    if not isinstance(planned_global_nets, list):
        print_log(LogColor.ERROR, "Phase0 Warning", "initial_global_nets 不是列表，已回退为空列表。")
        planned_global_nets = []
    phase0_plan["initial_global_nets"] = planned_global_nets

    if not planned_global_nets:
        print_log(
            LogColor.SYSTEM,
            "Phase0 Note",
            "Phase0 未提供 initial_global_nets，初始化阶段不会预置全局网络；可在后续动态补充。",
        )

    print_log(LogColor.SYSTEM, "Design Node", "Phase0 完成，正在构建初始骨架图...")
    initial_graph = build_initial_skeleton_graph(phase0_plan, component_catalog=component_list)
    structure_errors = validate_graph_structure(initial_graph)
    if structure_errors:
        preview = "\n".join(structure_errors[:5])
        print_log(LogColor.ERROR, "Structure Warning", f"初始骨架检测到结构问题（仅展示前5条）:\n{preview}")

    set_current_graph(initial_graph)
    G = get_current_graph()
    modules = _sort_modules_by_macro_dependencies(phase0_plan)
    print_log(LogColor.SYSTEM, "Design Node", f"待处理模块：{modules}")

    module_specs: Dict[str, Dict[str, Any]] = {}
    for module in phase0_plan.get("modules", []):
        if not isinstance(module, dict):
            continue
        module_id = str(module.get("module_id", "")).strip()
        if module_id:
            module_specs[module_id] = module

    shared_nets = _build_shared_nets_lookup(phase0_plan, G)
    global_power_ground_nets = _collect_global_power_ground_net_names(phase0_plan)
    phase1_system_prompt = _build_phase1_component_prompt()

    process_metrics: Dict[str, Any] = {
        "erc_correction_rounds": 0,
        "per_component_retries": {},
        "failed_components": [],
        "phase0_interface_failures": {},
    }
    phase1_max_retries = state.get("phase1_max_retries", 6)
    phase1_call_index = 1

    last_module = ""
    for current_mod in modules:
        last_module = current_mod
        print_log(LogColor.SYSTEM, "Design Node", f"进入模块 '{current_mod}' 执行逐器件连线...")

        module_cfg = module_specs.get(current_mod, {})
        raw_components = module_cfg.get("components", []) if isinstance(module_cfg, dict) else []
        module_components = [
            str(c).strip()
            for c in raw_components
            if str(c).strip()
        ] if isinstance(raw_components, list) else []

        required_virtual_nets = _derive_module_required_virtual_nets(
            current_mod,
            module_cfg,
            shared_nets,
            global_power_ground_nets=global_power_ground_nets,
        )
        if required_virtual_nets:
            _materialize_module_virtual_nets(G, current_mod, required_virtual_nets, shared_nets)
            print_log(
                LogColor.SYSTEM,
                "Phase1 Contract",
                f"模块 '{current_mod}' 需接入的虚拟接口网络: {required_virtual_nets}",
            )

        if not module_components:
            if required_virtual_nets:
                process_metrics.setdefault("phase0_interface_failures", {})[current_mod] = required_virtual_nets
                print_log(
                    LogColor.ERROR,
                    "Phase1 Contract",
                    f"模块 '{current_mod}' 无可处理器件，无法满足 Phase0 虚拟接口网络约束: {required_virtual_nets}",
                )
            print_log(LogColor.SYSTEM, "Phase1 Note", f"模块 '{current_mod}' 没有可处理器件，跳过。")
            continue

        print_log(LogColor.SYSTEM, "Phase1 Order", f"模块 '{current_mod}' 器件顺序: {module_components}")

        for comp_idx, refdes in enumerate(module_components):
            if not G.has_node(refdes) or G.nodes[refdes].get("type") != "Component":
                print_log(LogColor.ERROR, "Phase1 Error", f"器件 '{refdes}' 不存在，跳过。")
                process_metrics["per_component_retries"][refdes] = phase1_max_retries
                process_metrics["failed_components"].append(refdes)
                continue

            owner = _component_module_id(G, refdes)
            if owner != current_mod:
                print_log(
                    LogColor.ERROR,
                    "Phase1 Error",
                    f"器件 '{refdes}' 所属模块为 '{owner}'，与当前模块 '{current_mod}' 不一致。",
                )
                process_metrics["per_component_retries"][refdes] = phase1_max_retries
                process_metrics["failed_components"].append(refdes)
                continue

            current_part_number = _component_part_number_from_graph(G, refdes)
            base_environment = _build_module_environment(
                G,
                current_mod,
                module_cfg,
                phase0_plan,
                shared_nets,
            )
            if required_virtual_nets:
                base_environment["virtual_nets_to_connect_now"] = _missing_required_virtual_nets_for_module(
                    G,
                    current_mod,
                    required_virtual_nets,
                )
                base_environment["virtual_nets_contract_note"] = (
                    "virtual_nets_to_connect_now 代表该模块必须立即接入的跨模块接口网络；"
                    "这些网络对应其他模块的真实连接需求，禁止延期到后续模块或后续轮次。"
                )
            current_component_payload = _build_component_input_payload(
                G,
                refdes,
                base_environment,
            )

            base_input = {
                "current_component": current_component_payload,
                "environment": base_environment,
            }

            previous_output: Dict[str, Any] = {
                "components": [
                    {
                        "refdes": refdes,
                        "part_number": current_part_number,
                    }
                ],
                "nets": [],
            }
            retry_feedback: List[Dict[str, Any]] = []

            accepted = False
            retry_count = 0
            max_retries = phase1_max_retries
            fallback_graph = None
            last_merge_errors: List[str] = []
            last_hard_erc_issues: List[ERCIssue] = []
            last_soft_erc_issues: List[ERCIssue] = []
            best_soft_graph = None
            best_soft_erc_issues: List[ERCIssue] = []
            best_soft_retry_count = 0

            while retry_count <= max_retries:
                if retry_count == 0:
                    llm_input = base_input
                else:
                    llm_input = {
                        "current_component": current_component_payload,
                        "environment": base_environment,
                        "your_previous_output": previous_output,
                        "erc_errors": retry_feedback,
                    }

                response = _call_llm(
                    client,
                    LLM_MODEL_NAME,
                    [
                        {"role": "system", "content": phase1_system_prompt},
                        {"role": "user", "content": json.dumps(llm_input, ensure_ascii=False, indent=2)},
                    ]
                )

                raw_phase1_output, _, _ = parse_and_print_response(
                    response,
                    phase="phase1",
                    module_id=current_mod,
                    loop_index=phase1_call_index,
                )
                phase1_call_index += 1

                try:
                    parsed_output = _extract_json_object(raw_phase1_output)
                except Exception as e:
                    retry_feedback = [
                        {
                            "rule": "OUTPUT_JSON_PARSE",
                            "target": refdes,
                            "severity": "error",
                            "message": f"Invalid JSON output: {e}",
                        }
                    ]
                    last_merge_errors = [f"Invalid JSON output: {e}"]
                    last_hard_erc_issues = []
                    last_soft_erc_issues = []
                    retry_count += 1
                    if retry_count <= max_retries:
                        print_log(
                            LogColor.ERROR,
                            "Phase1 Retry",
                            f"器件 '{refdes}' 输出 JSON 解析失败，重试 {retry_count}/{max_retries}。",
                        )
                        print_log(
                            LogColor.ERROR,
                            "Phase1 Retry Detail",
                            f"merge_errors:\n- Invalid JSON output: {e}",
                        )
                    continue

                normalized_output = _normalize_connection_output(
                    parsed_output,
                    current_refdes=refdes,
                    current_part_number=current_part_number,
                )
                previous_output = normalized_output

                G_trial = copy.deepcopy(G)
                _, merge_errors = _merge_component_output_to_graph(
                    G_trial,
                    module_id=current_mod,
                    current_refdes=refdes,
                    output=normalized_output,
                    shared_nets=shared_nets,
                )

                hard_erc_issues: List[ERCIssue] = []
                soft_issues: List[ERCIssue] = []
                if not merge_errors:
                    erc_issues = _run_component_level_erc(G_trial, {refdes}, require_ready=False)
                    hard_erc_issues = [
                        i for i in erc_issues if str(i.get("severity", "warning")).lower() == "error"
                    ]
                    soft_issues = [
                        i for i in erc_issues if str(i.get("severity", "warning")).lower() != "error"
                    ]
                    fallback_graph = G_trial

                # 在模块最后一个核心器件处强制校验 Phase0 虚拟接口网络契约。
                if (
                    not merge_errors
                    and not hard_erc_issues
                    and comp_idx == len(module_components) - 1
                    and required_virtual_nets
                ):
                    missing_required = _missing_required_virtual_nets_for_module(
                        G_trial,
                        current_mod,
                        required_virtual_nets,
                    )
                    if missing_required:
                        merge_errors.extend(
                            [
                                (
                                    "Phase0 interface contract violation: "
                                    f"module '{current_mod}' must connect at least one local pin to interface net '{net_name}' now; "
                                    "this net is a cross-module contract with existing counterpart demand, deferring to later modules is not allowed."
                                )
                                for net_name in missing_required
                            ]
                        )

                last_merge_errors = list(merge_errors)
                last_hard_erc_issues = list(hard_erc_issues)
                last_soft_erc_issues = list(soft_issues)

                if not merge_errors and not hard_erc_issues and not soft_issues:
                    G = G_trial
                    set_current_graph(G)
                    process_metrics["per_component_retries"][refdes] = retry_count
                    process_metrics["erc_correction_rounds"] += retry_count

                    save_round_snapshot(G, current_mod, phase1_call_index, tag=f"component_{refdes}")
                    print_log(
                        LogColor.SYSTEM,
                        "Phase1",
                        f"器件 '{refdes}' 完成，重试次数={retry_count}。",
                    )
                    accepted = True
                    break

                if not merge_errors and not hard_erc_issues and soft_issues:
                    if (best_soft_graph is None) or (len(soft_issues) < len(best_soft_erc_issues)):
                        best_soft_graph = G_trial
                        best_soft_erc_issues = list(soft_issues)
                        best_soft_retry_count = retry_count

                    retry_feedback = _build_component_retry_feedback(
                        refdes,
                        [],
                        [],
                        soft_erc_issues=soft_issues,
                        best_effort_soft=True,
                    )
                    retry_count += 1

                    if retry_count <= max_retries:
                        print_log(
                            LogColor.SYSTEM,
                            "Phase1 Soft ERC",
                            f"器件 '{refdes}' 存在软违规，执行 best effort 修正（soft_erc={len(soft_issues)}），重试 {retry_count}/{max_retries}。",
                        )
                        soft_detail_lines = ["soft_erc_issues:"]
                        for issue in soft_issues[:10]:
                            soft_detail_lines.append(f"- {_format_erc_issue(issue)}")
                        print_log(
                            LogColor.SYSTEM,
                            "Phase1 Soft ERC Detail",
                            "\n".join(soft_detail_lines),
                        )
                        continue

                    chosen_graph = best_soft_graph if best_soft_graph is not None else G_trial
                    chosen_soft_issues = best_soft_erc_issues if best_soft_graph is not None else list(soft_issues)
                    chosen_retry_count = best_soft_retry_count if best_soft_graph is not None else max_retries

                    G = chosen_graph
                    set_current_graph(G)
                    process_metrics["per_component_retries"][refdes] = max_retries
                    process_metrics["erc_correction_rounds"] += max_retries

                    print_log(
                        LogColor.SYSTEM,
                        "Phase1 Soft ERC",
                        f"器件 '{refdes}' soft best effort 已到上限，保留最优版本（soft_erc={len(chosen_soft_issues)}，best_retry={chosen_retry_count}）。",
                    )
                    final_soft_lines = ["retained soft_erc_issues:"]
                    for issue in chosen_soft_issues[:15]:
                        final_soft_lines.append(f"- {_format_erc_issue(issue)}")
                    print_log(
                        LogColor.SYSTEM,
                        "Phase1 Soft ERC Detail",
                        "\n".join(final_soft_lines),
                    )

                    save_round_snapshot(G, current_mod, phase1_call_index, tag=f"component_{refdes}_best_effort")
                    print_log(
                        LogColor.SYSTEM,
                        "Phase1",
                        f"器件 '{refdes}' 完成（soft best effort），重试次数={max_retries}。",
                    )
                    accepted = True
                    break

                retry_feedback = _build_component_retry_feedback(
                    refdes,
                    merge_errors,
                    hard_erc_issues,
                    soft_erc_issues=soft_issues,
                    best_effort_soft=False,
                )
                retry_count += 1

                if retry_count <= max_retries:
                    print_log(
                        LogColor.ERROR,
                        "Phase1 Retry",
                        f"器件 '{refdes}' 需修正（merge_errors={len(merge_errors)}, hard_erc={len(hard_erc_issues)}, soft_erc={len(soft_issues)}），重试 {retry_count}/{max_retries}。",
                    )

                    detail_lines: List[str] = []
                    if merge_errors:
                        detail_lines.append("merge_errors:")
                        for err in merge_errors[:10]:
                            detail_lines.append(f"- {err}")
                    if hard_erc_issues:
                        detail_lines.append("hard_erc_issues:")
                        for issue in hard_erc_issues[:10]:
                            detail_lines.append(f"- {_format_erc_issue(issue)}")
                    if soft_issues:
                        detail_lines.append("soft_erc_issues:")
                        for issue in soft_issues[:10]:
                            detail_lines.append(f"- {_format_erc_issue(issue)}")
                    if detail_lines:
                        print_log(
                            LogColor.ERROR,
                            "Phase1 Retry Detail",
                            "\n".join(detail_lines),
                        )

            if not accepted:
                process_metrics["per_component_retries"][refdes] = max_retries
                process_metrics["failed_components"].append(refdes)
                if fallback_graph is not None:
                    G = fallback_graph
                    set_current_graph(G)
                print_log(
                    LogColor.ERROR,
                    "Phase1 Error",
                    f"器件 '{refdes}' 在 {max_retries} 次重试后仍未通过硬校验。",
                )

                final_detail_lines: List[str] = []
                if last_merge_errors:
                    final_detail_lines.append("last merge_errors:")
                    for err in last_merge_errors[:15]:
                        final_detail_lines.append(f"- {err}")
                if last_hard_erc_issues:
                    final_detail_lines.append("last hard_erc_issues:")
                    for issue in last_hard_erc_issues[:15]:
                        final_detail_lines.append(f"- {_format_erc_issue(issue)}")
                if last_soft_erc_issues:
                    final_detail_lines.append("last soft_erc_issues:")
                    for issue in last_soft_erc_issues[:15]:
                        final_detail_lines.append(f"- {_format_erc_issue(issue)}")
                if final_detail_lines:
                    print_log(
                        LogColor.ERROR,
                        "Phase1 Error Detail",
                        "\n".join(final_detail_lines),
                    )

        if required_virtual_nets:
            missing_after_module = _missing_required_virtual_nets_for_module(
                G,
                current_mod,
                required_virtual_nets,
            )
            if missing_after_module:
                process_metrics.setdefault("phase0_interface_failures", {})[current_mod] = missing_after_module
                print_log(
                    LogColor.ERROR,
                    "Phase1 Contract",
                    (
                        f"模块 '{current_mod}' 未满足 Phase0 虚拟接口网络约束，缺失网络: {missing_after_module}. "
                        "将于编译前硬阻断。"
                    ),
                )

        print_log(LogColor.SYSTEM, "Phase1", f"模块 '{current_mod}' 处理完成。")

    clear_active_module_scope()

    passive_component_count = _count_existing_passive_components(G)
    process_metrics["existing_passive_components_count"] = passive_component_count
    if passive_component_count > 0:
        print_log(
            LogColor.SYSTEM,
            "Phase1 Summary",
            f"存在无源器件数量: {passive_component_count}",
        )

    if process_metrics["failed_components"]:
        print_log(
            LogColor.ERROR,
            "Phase1 Summary",
            f"存在未收敛器件: {process_metrics['failed_components']}",
        )

    if process_metrics.get("phase0_interface_failures"):
        print_log(
            LogColor.ERROR,
            "Phase1 Summary",
            f"存在 Phase0 接口未落实模块: {process_metrics['phase0_interface_failures']}",
        )

    return {
        "phase0_plan": phase0_plan,
        "active_module": last_module,
        "completed_modules": modules,
        "process_metrics": process_metrics,
    }


def _merge_macro_connection_alias_nets_final(
    G,
    phase0_plan: Dict[str, Any],
) -> Dict[str, int]:
    """
    在编译前执行最终别名合并：
    - 对每条 macro_connection，将 source/target 端口名对应的网络并到 net_name。
    - GND 相关网络不自动合并，保留后续单点桥接空间。
    """
    stats = {
        "processed": 0,
        "skipped_ground": 0,
        "created_canonical": 0,
        "merged_alias_nodes": 0,
        "skipped_invalid": 0,
    }

    if not isinstance(phase0_plan, dict):
        return stats

    raw_macro = phase0_plan.get("macro_connections", [])
    if not isinstance(raw_macro, list):
        return stats

    def _normalize_conn(conn: Dict[str, Any], index: int) -> Dict[str, str]:
        source = str(conn.get("source", conn.get("source_port", ""))).strip()
        target = str(conn.get("target", conn.get("target_port", ""))).strip()
        net_name = str(conn.get("net_name", "")).strip()
        if not net_name:
            net_name = _port_name_from_endpoint(source) or _port_name_from_endpoint(target)
        if not net_name:
            net_name = f"AUTO_NET_{index + 1}"
        return {
            "source": source,
            "target": target,
            "net_name": net_name,
        }

    def _ensure_global_net(net_name: str) -> bool:
        if G.has_node(net_name):
            if G.nodes[net_name].get("type") != "Net":
                return False
        else:
            inferred = _normalize_net_type(net_name)
            G.add_node(
                net_name,
                type="Net",
                name=net_name,
                scope="Global",
                net_type=inferred,
                net_class=inferred,
                semantic_role="Phase0 macro canonical net",
                source_pin="",
            )
            stats["created_canonical"] += 1

        G.nodes[net_name]["scope"] = "Global"
        if not str(G.nodes[net_name].get("name", "")).strip():
            G.nodes[net_name]["name"] = net_name

        if not str(G.nodes[net_name].get("net_type", "")).strip():
            inferred = _normalize_net_type(net_name)
            G.nodes[net_name]["net_type"] = inferred

        if not str(G.nodes[net_name].get("net_class", "")).strip():
            G.nodes[net_name]["net_class"] = str(G.nodes[net_name].get("net_type", "Signal"))

        # 转为 Global 后，不再保留 Local Net -> Module 归属边。
        for nbr in list(G.neighbors(net_name)):
            edge_data = G.get_edge_data(net_name, nbr) or {}
            if edge_data.get("relation") == "Belongs_To" and G.nodes[nbr].get("type") == "Module":
                G.remove_edge(net_name, nbr)

        return True

    for idx, raw_conn in enumerate(raw_macro):
        if not isinstance(raw_conn, dict):
            continue

        conn = _normalize_conn(raw_conn, idx)
        source = conn["source"]
        target = conn["target"]
        net_name = conn["net_name"]

        src_port = _port_name_from_endpoint(source)
        tgt_port = _port_name_from_endpoint(target)

        # Ground 相关网络不自动归并，留给后续桥接策略处理。
        if _is_ground_like_name(net_name) or _is_ground_like_name(src_port) or _is_ground_like_name(tgt_port):
            stats["skipped_ground"] += 1
            continue

        alias_names = {
            str(src_port).strip(),
            str(tgt_port).strip(),
        }
        alias_names = {n for n in alias_names if n}
        if not alias_names:
            continue

        if not _ensure_global_net(net_name):
            print_log(
                LogColor.ERROR,
                "Phase4 Merge Warning",
                f"macro_connections[{idx}] 目标网络 '{net_name}' 已存在且不是 Net，跳过该条合并。",
            )
            stats["skipped_invalid"] += 1
            continue

        stats["processed"] += 1

        for alias in sorted(alias_names):
            if alias == net_name:
                continue
            if not G.has_node(alias):
                continue
            if G.nodes[alias].get("type") != "Net":
                print_log(
                    LogColor.ERROR,
                    "Phase4 Merge Warning",
                    f"别名网络 '{alias}' 存在但不是 Net，跳过与 '{net_name}' 的合并。",
                )
                stats["skipped_invalid"] += 1
                continue

            for nbr in list(G.neighbors(alias)):
                edge_data = G.get_edge_data(alias, nbr) or {}
                if edge_data.get("relation") != "Connected_To":
                    continue
                if G.nodes[nbr].get("type") != "Pin":
                    continue
                if not G.has_edge(nbr, net_name):
                    G.add_edge(nbr, net_name, relation="Connected_To")

            if G.has_node(alias):
                G.remove_node(alias)
                stats["merged_alias_nodes"] += 1

    return stats

# ==========================================
# 节点 2：编译器 (Compiler Node)
# ==========================================
def compiler_node(state: AgentState) -> AgentState:
    print_log(LogColor.SYSTEM, "Compiler Node", "所有模块 DFS 完成。开始进行全局缝合与校验（暂不生成 SKiDL）...")

    clear_active_module_scope()
    G = get_current_graph()
    phase0_plan = state.get("phase0_plan", {}) if isinstance(state.get("phase0_plan", {}), dict) else {}

    def _finalize(result: Dict[str, Any]) -> AgentState:
        artifacts = export_final_graph_artifacts(G, state, result)
        if artifacts:
            result["artifacts"] = artifacts
        return {
            "compile_result": result,
            "skidl_code": "",
        }

    merge_stats = _merge_macro_connection_alias_nets_final(G, phase0_plan)
    if any(int(merge_stats.get(k, 0)) > 0 for k in ["processed", "merged_alias_nodes", "created_canonical", "skipped_ground"]):
        print_log(
            LogColor.SYSTEM,
            "Phase4 Macro Merge",
            (
                "macro_connections 末尾归并统计: "
                f"processed={merge_stats.get('processed', 0)}, "
                f"merged_alias_nodes={merge_stats.get('merged_alias_nodes', 0)}, "
                f"created_canonical={merge_stats.get('created_canonical', 0)}, "
                f"skipped_ground={merge_stats.get('skipped_ground', 0)}, "
                f"skipped_invalid={merge_stats.get('skipped_invalid', 0)}"
            ),
        )

    process_metrics = state.get("process_metrics", {}) if isinstance(state.get("process_metrics", {}), dict) else {}
    interface_failures = process_metrics.get("phase0_interface_failures", {})
    if isinstance(interface_failures, dict) and interface_failures:
        lines = ["编译前接口契约检查失败：以下模块未完成 Phase0 规划的虚拟接口网络连线。"]
        for mod, nets in interface_failures.items():
            if isinstance(nets, list):
                lines.append(f"- {mod}: missing {sorted(str(n) for n in nets)}")
            else:
                lines.append(f"- {mod}: missing {nets}")
        report = "\n".join(lines)
        print_log(LogColor.ERROR, "Phase0 Interface Gate", report)
        return _finalize(
            {
                "status": "blocked_phase0_interface_contract",
                "message": "Phase0 interface contract violations detected before compile.",
                "phase0_interface_failures": interface_failures,
            }
        )

    # 编译前强闭环：所有组件的局部 A/B 类硬违规必须归零。
    all_components = {n for n, attr in G.nodes(data=True) if attr.get("type") == "Component"}
    local_issues = _run_component_level_erc(G, all_components, require_ready=False)
    local_hard = [i for i in local_issues if str(i.get("severity", "warning")).lower() == "error"]
    local_soft = [i for i in local_issues if str(i.get("severity", "warning")).lower() != "error"]
    if local_hard:
        lines = ["编译前局部 ERC（A/B 类）仍存在硬违规，已阻断 SKiDL 输出："]
        for issue in local_hard[:40]:
            lines.append(f"- {_format_erc_issue(issue)}")
        if local_soft:
            lines.append("附加软违规提示：")
            for issue in local_soft[:10]:
                lines.append(f"- {_format_erc_issue(issue)}")

        report = "\n".join(lines)
        print_log(LogColor.ERROR, "Local ERC Hard Gate", report)
        return _finalize(
            {
                "status": "blocked_local_erc",
                "message": "Local ERC hard violations detected before compile.",
                "local_hard_count": len(local_hard),
                "local_soft_count": len(local_soft),
            }
        )

    pre_errors = validate_graph_structure(G)
    if pre_errors:
        preview = "\n".join(pre_errors[:5])
        print_log(LogColor.ERROR, "Pre-Compile Structure Warning", f"编译前检测到结构问题（前5条）:\n{preview}")

    pre_stitch_stats = {
        "loops": 0,
        "tool_calls": 0,
        "successful_calls": 0,
    }
    global_repair_stats = {
        "loops": 0,
        "tool_calls": 0,
        "successful_calls": 0,
    }

    has_virtual_ports = any(attr.get("type") == "VirtualPort" for _, attr in G.nodes(data=True))
    if has_virtual_ports:
        stitch_msg = stitch_global_modules(G)
        print_log(LogColor.SYSTEM, "全局缝合", stitch_msg)
        if str(stitch_msg).startswith("Error"):
            return _finalize(
                {
                    "status": "blocked_stitch",
                    "message": str(stitch_msg),
                }
            )
    else:
        print_log(LogColor.SYSTEM, "全局缝合", "未检测到 VirtualPort，跳过缝合阶段。")

    post_errors = validate_graph_structure(G)
    if post_errors:
        preview = "\n".join(post_errors[:5])
        print_log(LogColor.ERROR, "Post-Stitch Structure Warning", f"缝合后检测到结构问题（前5条）:\n{preview}")

    # 全局阶段：在 Phase4 缝合后执行 C 类 ERC（属性传播型）检查。
    global_erc_issues = _run_global_level_erc(G)
    global_hard = [i for i in global_erc_issues if str(i.get("severity", "warning")).lower() == "error"]
    global_soft = [i for i in global_erc_issues if str(i.get("severity", "warning")).lower() != "error"]

    if global_hard:
        report = _format_global_erc_report(global_hard, global_soft)
        print_log(LogColor.ERROR, "Global ERC", report)
        return _finalize(
            {
                "status": "blocked_global_erc",
                "message": "Global ERC hard violations detected.",
                "global_hard_count": len(global_hard),
                "global_soft_count": len(global_soft),
                "pre_stitch_loops": int(pre_stitch_stats.get("loops", 0)),
                "pre_stitch_tool_calls": int(pre_stitch_stats.get("tool_calls", 0)),
                "pre_stitch_successful_calls": int(pre_stitch_stats.get("successful_calls", 0)),
                "global_repair_loops": int(global_repair_stats.get("loops", 0)),
                "global_repair_tool_calls": int(global_repair_stats.get("tool_calls", 0)),
                "global_repair_successful_calls": int(global_repair_stats.get("successful_calls", 0)),
            }
        )

    if global_soft:
        lines = ["全局 ERC（C 类）软违规提示（未阻断编译）："]
        for issue in global_soft[:20]:
            lines.append(f"- {_format_erc_issue(issue)}")
        print_log(LogColor.SYSTEM, "Global ERC", "\n".join(lines))
    
    compile_result = {
        "status": "ok",
        "message": "Validation and stitching completed.",
        "local_hard_count": len(local_hard),
        "local_soft_count": len(local_soft),
        "global_hard_count": len(global_hard),
        "global_soft_count": len(global_soft),
        "pre_stitch_loops": int(pre_stitch_stats.get("loops", 0)),
        "pre_stitch_tool_calls": int(pre_stitch_stats.get("tool_calls", 0)),
        "pre_stitch_successful_calls": int(pre_stitch_stats.get("successful_calls", 0)),
        "global_repair_loops": int(global_repair_stats.get("loops", 0)),
        "global_repair_tool_calls": int(global_repair_stats.get("tool_calls", 0)),
        "global_repair_successful_calls": int(global_repair_stats.get("successful_calls", 0)),
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
    }
    _append_runtime_log("COMPILE_RESULT", json.dumps(compile_result, ensure_ascii=False))
    print_log(LogColor.SYSTEM, "Compiler Node", "全局缝合与校验完成（SKiDL 生成已关闭）。")

    return _finalize(compile_result)

# ==========================================
# 构建并编译 LangGraph
# ==========================================
def build_agent_graph():
    if StateGraph is None:
        print_log(LogColor.SYSTEM, "Graph Backend", "未检测到 langgraph，使用顺序执行器（design -> compiler）。")

        class SequentialWorkflow:
            def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
                state: Dict[str, Any] = dict(initial_state)
                state.update(design_node(state))
                state.update(compiler_node(state))
                return state

        return SequentialWorkflow()

    workflow = StateGraph(AgentState)

    workflow.add_node("design", design_node)
    workflow.add_node("compiler", compiler_node)

    workflow.add_edge(START, "design")
    workflow.add_edge("design", "compiler")

    workflow.add_edge("compiler", END)

    return workflow.compile()

# ==========================================
# 测试入口
# ==========================================
if __name__ == "__main__":
    app = build_agent_graph()

    try:
        if len(sys.argv) >= 2:
            case_index = int(sys.argv[1])
        else:
            raw = input("请输入测试序号（1-based）: ").strip()
            if not raw:
                raise ValueError("未提供测试序号")
            case_index = int(raw)

        case = get_benchmark_case_by_index(case_index)
        initial_state = build_initial_state_from_case_index(case_index)
    except Exception as e:
        print_log(LogColor.ERROR, "Case Selection", str(e))
        print("用法: python agent.py <测试序号>  # 例如 python agent.py 1")
        sys.exit(1)
    
    print("\n" + "="*50)
    print_log(LogColor.SYSTEM, "INIT", "OpenEDA Agent Workflow Started")
    print_log(
        LogColor.SYSTEM,
        "CASE",
        f"#{case_index} | {case.get('design_id', '')} | {case.get('title', '')}",
    )
    print("="*50 + "\n")
    
    final_state = app.invoke(initial_state)
    
    print("\n" + "="*50)
    print_log(LogColor.OUTPUT, "SUCCESS", "流程完成，输出编译阶段结果如下：")
    print("="*50)
    print(json.dumps(final_state.get("compile_result", {}), ensure_ascii=False, indent=2))