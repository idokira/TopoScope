from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

 
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Allow running as a script: python benchmark/runners/run_full_system.py
if __package__ in {None, ""}:
	if str(PROJECT_ROOT) not in sys.path:
		sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.eval.graph_utils import extract_connection_set_raw, serialize_graph
from benchmark.runners.llm_clients import build_client, list_providers, load_provider_config


DEFAULT_CASES_FILE = PROJECT_ROOT / "benchmark/cases/benchmark_cases.json"
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "benchmark/configs/llm_clients.json"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "benchmark/results/full_system"


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


def _load_cases(cases_file: Path) -> List[Dict[str, Any]]:
	if not cases_file.exists():
		raise FileNotFoundError(f"Cases file not found: {cases_file}")

	with cases_file.open("r", encoding="utf-8") as f:
		data = json.load(f)

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


def _disable_agent_file_artifacts(pipeline_agent: Any) -> None:
	"""
	Keep terminal prints, disable file snapshots and per-round response logs.
	"""

	def _disabled_init_run_logging(_task_description: str, extra_meta: Optional[Dict[str, Any]] = None) -> str:
		_ = extra_meta
		pipeline_agent.RUN_LOG_DIR = None
		return "disabled"

	pipeline_agent.RUN_LOG_DIR = None
	pipeline_agent.init_run_logging = _disabled_init_run_logging
	pipeline_agent.save_model_response_artifacts = lambda *args, **kwargs: None
	pipeline_agent.save_round_snapshot = lambda *args, **kwargs: None
	pipeline_agent.export_final_graph_artifacts = lambda *args, **kwargs: {}


def _build_initial_state(case: Dict[str, Any]) -> Dict[str, Any]:
	return {
		"design_id": str(case.get("design_id", "")).strip(),
		"task_description": str(case.get("task_description", "")).strip(),
		"component_list": case.get("component_list", []),
	}


def _save_result_json(
	case: Dict[str, Any],
	provider: str,
	model_name: str,
	run_id: int,
	graph: Any,
	final_state: Dict[str, Any],
	wall_time_seconds: float,
	results_root: Path,
) -> Path:
	design_id = str(case.get("design_id", "")).strip()
	tier = case.get("tier")
	design_type = case.get("type")

	connection_set = extract_connection_set_raw(graph)
	graph_snapshot = serialize_graph(graph)

	model_dir = results_root / _safe_path_token(model_name)
	design_dir = model_dir / _safe_path_token(design_id)
	design_dir.mkdir(parents=True, exist_ok=True)

	out_file = design_dir / f"{design_id}_run{run_id}.json"

	result: Dict[str, Any] = {
		"meta": {
			"design_id": design_id,
			"tier": tier,
			"type": design_type,
			"provider": provider,
			"model": model_name,
			"method": "full_system",
			"run_id": run_id,
			"wall_time_seconds": round(wall_time_seconds, 3),
		},
		"compile_result": final_state.get("compile_result", {}),
		"connection_set": connection_set,
		"graph_snapshot": graph_snapshot,
		"process_metrics": final_state.get("process_metrics", {}),
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
) -> Path:
	try:
		import agent as pipeline_agent
		from tools.graph_action import get_current_graph
	except ModuleNotFoundError as exc:
		raise RuntimeError(
			f"Missing runtime dependency for full-system pipeline: {exc}. "
			"Please install required packages (e.g., langchain_core)."
		) from exc

	llm_client = build_client(provider=provider, config_path=str(config_file))

	pipeline_agent.client = llm_client.raw_client
	pipeline_agent.LLM_MODEL_NAME = llm_client.model
	_disable_agent_file_artifacts(pipeline_agent)

	app = pipeline_agent.build_agent_graph()
	initial_state = _build_initial_state(case)

	design_id = str(case.get("design_id", "")).strip()
	print("=" * 60)
	print(f"[RUN] provider={llm_client.provider} model={llm_client.model} case={design_id}")
	print("=" * 60)

	t0 = time.time()
	old_cwd = Path.cwd()
	try:
		# agent.py internally reads prompt files by relative paths; run from repo root for stability.
		os.chdir(str(PROJECT_ROOT))
		final_state = app.invoke(initial_state)
	finally:
		os.chdir(str(old_cwd))
	elapsed = time.time() - t0
	graph = get_current_graph()

	model_dir = results_root / _safe_path_token(llm_client.model)
	design_dir = model_dir / _safe_path_token(design_id)
	actual_run_id = run_id if run_id is not None else _next_run_id(design_dir, design_id)
	out_file = _save_result_json(
		case=case,
		provider=llm_client.provider,
		model_name=llm_client.model,
		run_id=actual_run_id,
		graph=graph,
		final_state=final_state,
		wall_time_seconds=elapsed,
		results_root=results_root,
	)

	print("\n" + "=" * 60)
	print("[DONE] compile_result:")
	print(json.dumps(final_state.get("compile_result", {}), ensure_ascii=False, indent=2))
	print(f"[DONE] saved result -> {out_file}")
	print("=" * 60)

	return out_file


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run full-system pipeline on benchmark cases.")
	parser.add_argument(
		"case",
		nargs="?",
		default=None,
		help="Optional case selector: 1-based index (e.g. 3) or design id (e.g. T2-1)",
	)
	parser.add_argument("--provider", default=None, help="Provider key in llm_clients config (e.g. qwen)")
	parser.add_argument("--config", default=str(DEFAULT_CONFIG_FILE), help="Path to llm client config JSON")
	parser.add_argument("--cases", default=str(DEFAULT_CASES_FILE), help="Path to benchmark cases JSON")
	parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT), help="Root output directory")
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
	)


if __name__ == "__main__":
	main()
