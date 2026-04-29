from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Allow running as a script: python benchmark/runners/run_baseline.py
if __package__ in {None, ""}:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.eval.graph_utils import build_graph_from_connection_set, serialize_graph
from benchmark.runners.llm_clients import build_client, list_providers, load_provider_config
from tools.erc_verification import ERCIssue, ERCVerificationEngine


def _format_erc_issue(issue: ERCIssue) -> str:
    severity = str(issue.get("severity", "warning")).lower()
    sev_tag = "ERROR" if severity == "error" else "WARNING"
    rule_type = issue.get("rule_type", "UNKNOWN_RULE")
    message = issue.get("message", "")
    description = issue.get("description", "")
    return f"[{sev_tag}] {rule_type}: {message} (Rule: {description})"


def _get_component_rules_from_db(G, comp_id: str, db: List[Dict[str, Any]]):
    if not G.has_node(comp_id):
        return []

    part_number = G.nodes[comp_id].get("part_number")
    if not part_number:
        return []

    for item in db:
        if item.get("name") == part_number:
            rules = item.get("erc_rules", [])
            if isinstance(rules, list):
                return rules
            return []

    return []


def _run_all_erc(G, component_ids: set, db: List[Dict[str, Any]]) -> List[ERCIssue]:
    engine = ERCVerificationEngine(G)
    violations = []

    for comp_id in component_ids:
        violations.extend(engine.verify_pin_multiplexing(comp_id))

        rules = _get_component_rules_from_db(G, comp_id, db)
        if not rules:
            continue

        for rule in rules:
            if ERCVerificationEngine.is_local_rule(rule) or ERCVerificationEngine.is_global_rule(rule):
                violations.extend(engine.verify_rule(comp_id, rule))

    return violations


DEFAULT_CASES_FILE = PROJECT_ROOT / "benchmark/cases/benchmark_cases.json"
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "benchmark/configs/llm_clients.json"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "benchmark/results/ablation/erc_only"

BASE_PROMPT_FILE = PROJECT_ROOT / "prompts/baseline_prompt.md"
BASE_KNOWLEDGE_FILE = PROJECT_ROOT / "prompts/base_design_knowledge.md"
NAMING_RULES_FILE = PROJECT_ROOT / "prompts/naming_rules_prompt.md"
COMPONENT_DB_FILE = PROJECT_ROOT / "benchmark/component_data/Extracted_Components.json"


def _resolve_path_arg(path_value: str) -> Path:
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p

    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    root_candidate = (PROJECT_ROOT / p).resolve()
    if root_candidate.exists():
        return root_candidate

    return cwd_candidate


def _load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def _load_json_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_cases(cases_file: Path) -> List[Dict[str, Any]]:
    data = _load_json_file(cases_file)
    if not isinstance(data, list):
        raise ValueError(f"Invalid cases format in {cases_file}: expected list")
    return data


def _get_case_by_index(cases: List[Dict[str, Any]], case_index: int) -> Dict[str, Any]:
    if case_index < 1 or case_index > len(cases):
        raise IndexError(f"Case index out of range: {case_index}. Valid range: 1..{len(cases)}")
    return cases[case_index - 1]


def _get_case_by_design_id(cases: List[Dict[str, Any]], design_id: str) -> Dict[str, Any]:
    target = design_id.strip()
    for case in cases:
        if str(case.get("design_id", "")).strip() == target:
            return case
    raise KeyError(f"Design id not found in cases: {design_id}")


def _select_case(
    cases: List[Dict[str, Any]],
    positional_case: Optional[str],
    case_index: Optional[int],
    design_id: Optional[str],
) -> Dict[str, Any]:
    if positional_case is not None and (case_index is not None or design_id):
        raise ValueError("Please provide either positional CASE or --case-index/--design-id, not both.")

    if positional_case is not None:
        selector = positional_case.strip()
        if not selector:
            raise ValueError("Positional CASE is empty.")
        if selector.isdigit():
            return _get_case_by_index(cases, int(selector))
        return _get_case_by_design_id(cases, selector)

    if case_index is not None and design_id:
        raise ValueError("Please provide only one of --case-index or --design-id.")

    if design_id:
        return _get_case_by_design_id(cases, design_id)

    selected_index = case_index if case_index is not None else 1
    return _get_case_by_index(cases, selected_index)


def _safe_path_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())


def _next_run_id(design_dir: Path, design_id: str) -> int:
    if not design_dir.exists():
        return 1

    pattern = re.compile(rf"^{re.escape(design_id)}_run(\d+)\.json$")
    max_id = 0
    for fp in design_dir.glob(f"{design_id}_run*.json"):
        m = pattern.match(fp.name)
        if m:
            max_id = max(max_id, int(m.group(1)))

    return max_id + 1


def _load_base_knowledge() -> str:
    base_knowledge = _load_text(BASE_KNOWLEDGE_FILE)
    naming_rules = _load_text(NAMING_RULES_FILE)

    if "{naming_rules}" in base_knowledge:
        return base_knowledge.replace("{naming_rules}", naming_rules)

    return (
        base_knowledge
        + "\n\n---\n\n"
        + "## Naming Rules (Strictly Enforced)\n"
        + naming_rules
    )


def _load_component_db(db_file: Path) -> List[Dict[str, Any]]:
    data = _load_json_file(db_file)
    if isinstance(data, list):
        return data
    return []


def _search_component_doc(db: List[Dict[str, Any]], part_number: str) -> Optional[Dict[str, Any]]:
    q = part_number.strip().lower()
    if not q:
        return None

    for item in db:
        pn = str(item.get("part_number", "")).strip().lower()
        if pn == q:
            return item

    for item in db:
        pn = str(item.get("part_number", "")).strip().lower()
        if q and q in pn:
            return item

    return None


def _build_pins_docs(component_list: List[Dict[str, Any]], db: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []

    for comp in component_list:
        refdes = str(comp.get("refdes", "")).strip()
        part_number = str(comp.get("part_number", "")).strip()
        desc = str(comp.get("description", "")).strip()

        item = _search_component_doc(db, part_number)
        if not item:
            docs.append(
                {
                    "refdes": refdes,
                    "part_number": part_number,
                    "description": desc,
                    "pins": [],
                    "note": "Component data not found in benchmark/component_data/Extracted_Components.json",
                }
            )
            continue

        pins = []
        for pin in item.get("pins", []):
            pins.append(
                {
                    "number": pin.get("number"),
                    "name": pin.get("name"),
                    "electrical_type": pin.get("electrical_type", pin.get("type", "")),
                    "description": pin.get("description", ""),
                }
            )

        docs.append(
            {
                "refdes": refdes,
                "part_number": part_number,
                "description": desc,
                "pins": pins,
                "reference_circuits": item.get("reference_circuits", []),
            }
        )

    return docs


def _build_system_prompt(case: Dict[str, Any], pins_docs: List[Dict[str, Any]]) -> str:
    template = _load_text(BASE_PROMPT_FILE)
    base_knowledge = _load_base_knowledge()

    component_list = case.get("component_list", [])
    task_description = str(case.get("task_description", "")).strip()

    merged = template.replace("{base_design_knowledge}", base_knowledge)
    merged = merged.replace("{task_description}", task_description)
    merged = merged.replace(
        "{component_list}",
        json.dumps(component_list, ensure_ascii=False, indent=2),
    )
    merged = merged.replace(
        "{retrieved_component_docs}",
        json.dumps(pins_docs, ensure_ascii=False, indent=2),
    )
    return merged


def _response_text(response: Any) -> str:
    texts: List[str] = []

    output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            item_type = getattr(item, "type", "")
            if item_type == "message":
                content = getattr(item, "content", [])
                for c in content:
                    text = getattr(c, "text", "")
                    if text:
                        texts.append(text)

    if texts:
        return "\n".join(texts).strip()

    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        if hasattr(msg, "content") and msg.content:
            return msg.content.strip()

    maybe_text = getattr(response, "output_text", "")
    if isinstance(maybe_text, str) and maybe_text.strip():
        return maybe_text.strip()

    return str(response)


def _extract_json_from_text(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data, cleaned
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(cleaned[idx:])
            if isinstance(obj, dict):
                return obj, cleaned[idx : idx + end]
        except Exception:
            continue

    return None, None


def _extract_connection_set_payload(parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}

    if isinstance(parsed.get("connection_set"), dict):
        return parsed["connection_set"]

    for key, value in parsed.items():
        norm = str(key).strip().lower().replace(" ", "_")
        if norm == "connection_set" and isinstance(value, dict):
            return value

    if "components" in parsed or "nets" in parsed:
        return parsed

    return {}


def _normalize_connection_set(parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = _extract_connection_set_payload(parsed)

    comps_raw = payload.get("components", []) if isinstance(payload, dict) else []
    nets_raw = payload.get("nets", []) if isinstance(payload, dict) else []

    comps: List[Dict[str, str]] = []
    if isinstance(comps_raw, list):
        for item in comps_raw:
            if not isinstance(item, dict):
                continue
            refdes = str(item.get("refdes", "")).strip()
            part_number = str(item.get("part_number", "")).strip()
            if not refdes:
                continue
            comps.append({"refdes": refdes, "part_number": part_number})

    nets: List[Dict[str, Any]] = []
    if isinstance(nets_raw, list):
        for item in nets_raw:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name", "")).strip()
            pins_obj = item.get("pins", [])
            if isinstance(pins_obj, str):
                pins = [pins_obj.strip()] if pins_obj.strip() else []
            elif isinstance(pins_obj, list):
                pins = [str(p).strip() for p in pins_obj if str(p).strip()]
            else:
                pins = []

            if not name and not pins:
                continue
            nets.append({"name": name, "pins": pins})

    return {
        "components": comps,
        "nets": nets,
    }


def _save_result_json(
    case: Dict[str, Any],
    provider: str,
    model_name: str,
    run_id: int,
    prompt: str,
    parsed_json: Optional[Dict[str, Any]],
    wall_time_seconds: float,
    results_root: Path,
) -> Path:
    design_id = str(case.get("design_id", "")).strip()
    tier = case.get("tier")
    design_type = case.get("type")

    model_dir = results_root / _safe_path_token(model_name)
    design_dir = model_dir / _safe_path_token(design_id)
    design_dir.mkdir(parents=True, exist_ok=True)

    out_file = design_dir / f"{design_id}_run{run_id}.json"

    connection_set = _normalize_connection_set(parsed_json)
    graph = build_graph_from_connection_set(connection_set)
    graph_snapshot = serialize_graph(graph)

    result: Dict[str, Any] = {
        "meta": {
            "design_id": design_id,
            "tier": tier,
            "type": design_type,
            "provider": provider,
            "model": model_name,
            "method": "erc_only",
            "run_id": run_id,
            "wall_time_seconds": round(wall_time_seconds, 3),
        },
        "compile_result": parsed_json or {},
        "connection_set": connection_set,
        "graph_snapshot": graph_snapshot,
        "process_metrics": {},
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return out_file


def run_single_case(
    provider: str,
    config_file: Path,
    case: Dict[str, Any],
    run_id: Optional[int],
    results_root: Path,
    component_db_file: Path,
) -> Path:
    llm_client = build_client(provider=provider, config_path=str(config_file))

    db = _load_component_db(component_db_file)
    pins_docs = _build_pins_docs(case.get("component_list", []), db)
    prompt = _build_system_prompt(case, pins_docs)

    design_id = str(case.get("design_id", "")).strip()
    print("=" * 60)
    print(f"[RUN-ERC-ONLY] provider={llm_client.provider} model={llm_client.model} case={design_id}")
    print("=" * 60)

    t0 = time.time()

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": "Please directly output the final Connection Set JSON. Output JSON only.",
        },
    ]

    max_retries = 6
    retry_count = 0
    final_json = None
    best_soft_json = None
    min_warnings = float("inf")

    while retry_count <= max_retries:
        print(f"--- Iteration {retry_count} ---")
        response = llm_client.responses_create(input=messages)

        raw_text = _response_text(response)
        parsed_json, _ = _extract_json_from_text(raw_text)

        if not parsed_json:
            messages.append({"role": "assistant", "content": raw_text})
            messages.append(
                {
                    "role": "user",
                    "content": "No valid JSON object could be parsed. Please output a valid Connection Set JSON that follows the required schema.",
                }
            )
            retry_count += 1
            print(f"Retry {retry_count}: JSON parser error")
            continue

        connection_set = _normalize_connection_set(parsed_json)
        try:
            G = build_graph_from_connection_set(connection_set)
        except Exception as e:
            messages.append({"role": "assistant", "content": raw_text})
            messages.append(
                {
                    "role": "user",
                    "content": f"Failed to build the validation graph from the connection data: {e}. Please correct the netlist.",
                }
            )
            retry_count += 1
            print(f"Retry {retry_count}: graph build error")
            continue

        comp_ids = {n for n, attr in G.nodes(data=True) if attr.get("type") == "Component"}
        all_issues = _run_all_erc(G, comp_ids, db)

        hard_issues = [i for i in all_issues if str(i.get("severity", "")).lower() == "error"]
        soft_issues = [i for i in all_issues if str(i.get("severity", "")).lower() != "error"]

        if not hard_issues:
            final_json = parsed_json
            print("No hard ERC issues. Success.")
            break

        if len(soft_issues) < min_warnings:
            best_soft_json = parsed_json
            min_warnings = len(soft_issues)

        messages.append({"role": "assistant", "content": raw_text})

        fb_lines = ["The previous generated netlist contains the following ERC violations:"]
        for issue in hard_issues[:30]:
            fb_lines.append(f"- {_format_erc_issue(issue)}")

        if soft_issues:
            fb_lines.append("Additional suggestions. These are not mandatory, but they are recommended for review:")
            for issue in soft_issues[:10]:
                fb_lines.append(f"- {_format_erc_issue(issue)}")

        fb_lines.append(
            "Please review and correct the connection design based on the feedback above. "
            "Fix all [ERROR] violations, then output only the complete corrected Connection Set JSON."
        )

        feedback_text = "\n".join(fb_lines)
        messages.append({"role": "user", "content": feedback_text})

        retry_count += 1
        print(f"Retry {retry_count}: {len(hard_issues)} hard issues found, {len(soft_issues)} soft issues")

    if not final_json:
        final_json = best_soft_json or parsed_json

    elapsed = time.time() - t0

    model_dir = results_root / _safe_path_token(llm_client.model)
    design_dir = model_dir / _safe_path_token(design_id)
    actual_run_id = run_id if run_id is not None else _next_run_id(design_dir, design_id)

    out_file = _save_result_json(
        case=case,
        provider=llm_client.provider,
        model_name=llm_client.model,
        run_id=actual_run_id,
        prompt=prompt,
        parsed_json=final_json,
        wall_time_seconds=elapsed,
        results_root=results_root,
    )

    print("\n" + "=" * 60)
    print(f"[DONE] case={design_id} saved result -> {out_file}")
    print("=" * 60)

    return out_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ERC-only baseline generation on benchmark cases.")
    parser.add_argument(
        "case",
        nargs="?",
        default=None,
        help="Optional case selector: 1-based index, e.g. 3, or design id, e.g. T2-1",
    )
    parser.add_argument("--provider", default=None, help="Provider key in llm_clients config, e.g. qwen")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_FILE), help="Path to llm client config JSON")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_FILE), help="Path to benchmark cases JSON")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT), help="Root output directory")
    parser.add_argument("--component-db", default=str(COMPONENT_DB_FILE), help="Path to Extracted_Components.json")
    parser.add_argument("--case-index", type=int, default=None, help="1-based case index")
    parser.add_argument("--design-id", default=None, help="Run case by design id, e.g. T2-1")
    parser.add_argument("--run-id", type=int, default=None, help="Optional fixed run id")
    parser.add_argument("--list-providers", action="store_true", help="List providers from config and exit")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    config_path = _resolve_path_arg(args.config)
    cases_path = _resolve_path_arg(args.cases)
    results_root = _resolve_path_arg(args.results_root)
    component_db_path = _resolve_path_arg(args.component_db)

    if args.list_providers:
        providers = list_providers(config_path=str(config_path))
        print(json.dumps(sorted(providers.keys()), ensure_ascii=False, indent=2))
        return

    config = load_provider_config(provider=args.provider, config_path=str(config_path))
    provider = config.provider

    cases = _load_cases(cases_path)
    case = _select_case(
        cases=cases,
        positional_case=args.case,
        case_index=args.case_index,
        design_id=args.design_id,
    )

    run_single_case(
        provider=provider,
        config_file=config_path,
        case=case,
        run_id=args.run_id,
        results_root=results_root,
        component_db_file=component_db_path,
    )


if __name__ == "__main__":
    main()