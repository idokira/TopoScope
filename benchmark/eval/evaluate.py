from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Allow running as a script: python benchmark/eval/evaluate.py
if __package__ in {None, ""}:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.eval.assertion_checker import evaluate_assertions, list_multi_net_pin_violations
from benchmark.eval.export_tables import export_reports
from benchmark.eval.graph_utils import (
    build_graph_from_connection_set,
    deserialize_graph,
    extract_connection_set_raw,
)
from benchmark.eval.pass_checker import check_pass_single_with_graph, compute_pass_at_k
from benchmark.eval.pin_normalizer import normalize_connection_set


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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_str(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _normalize_design_id_token(token: str) -> str:
    text = _safe_str(token)
    if not text:
        return ""
    m = re.search(r"(T\d+-\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return text


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_assertion_gt_files(assertion_dir: Optional[Path]) -> Dict[str, Dict[str, object]]:
    """
    Load assertion ground-truth files from benchmark/ground_truth_assertion.
    """
    out: Dict[str, Dict[str, object]] = {}
    if assertion_dir is None or not assertion_dir.exists() or not assertion_dir.is_dir():
        return out

    for path in sorted(assertion_dir.glob("*.json")):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        design_id = _normalize_design_id_token(_safe_str(payload.get("design_id")))
        if not design_id:
            design_id = _normalize_design_id_token(path.stem)
        if not design_id:
            continue

        out[design_id] = payload

    return out


def _load_gt_files(assertion_dir: Optional[Path] = None) -> Dict[str, Dict[str, object]]:
    """
    Load evaluation ground truth exclusively from benchmark/ground_truth_assertion.
    """
    assertion_gt_map = _load_assertion_gt_files(assertion_dir)
    gt_files: Dict[str, Dict[str, object]] = {}

    if not assertion_gt_map:
        return gt_files

    for design_id, assertion_payload in sorted(assertion_gt_map.items()):
        if not isinstance(assertion_payload, dict):
            continue

        assertions = assertion_payload.get("verification_assertions", [])
        if not isinstance(assertions, list):
            print(
                f"[evaluate] Skip design '{design_id}': invalid assertion file (verification_assertions is not a list)."
            )
            continue

        gt_item: Dict[str, object] = {
            "design_id": design_id,
            "verification_assertions": [a for a in assertions if isinstance(a, dict)],
        }
        for field in ["tier", "type", "description"]:
            if field in assertion_payload:
                gt_item[field] = assertion_payload[field]

        gt_files[design_id] = gt_item

    return gt_files


def _discover_result_files(results_dir: Path) -> List[Path]:
    if results_dir.is_file():
        return [results_dir]
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob("*.json"))


def _cleanup_generated_task_summaries(results_path: Path) -> int:
    """
    Remove stale evaluation_summary.json files so current run outputs reflect strict matching.
    """
    if not results_path.exists() or results_path.is_file():
        return 0

    removed = 0
    for summary_path in results_path.rglob("evaluation_summary.json"):
        try:
            summary_path.unlink()
            removed += 1
        except Exception:
            continue
    return removed


def _infer_design_id(path: Path, payload: Dict[str, object]) -> Optional[str]:
    meta = payload.get("meta", {})
    if isinstance(meta, dict):
        did = _normalize_design_id_token(str(meta.get("design_id", "")).strip())
        if did:
            return did

    m = re.search(r"(T\d+-\d+)", path.stem)
    if m:
        return m.group(1).upper()

    return None


def _is_evaluable_result_payload(payload: Dict[str, object]) -> bool:
    if not isinstance(payload, dict):
        return False
    return isinstance(payload.get("graph_snapshot"), dict) or isinstance(payload.get("connection_set"), dict)


def _infer_method_model_ablation(
    path: Path,
    results_dir: Path,
    payload: Dict[str, object],
) -> Tuple[str, str, Optional[str], Optional[int]]:
    meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}

    method = str(meta.get("method", "")).strip()
    model = str(meta.get("model", "")).strip()
    ablation = str(meta.get("ablation", "")).strip() or None

    run_id_raw = meta.get("run_id")
    run_id: Optional[int] = None
    if isinstance(run_id_raw, int):
        run_id = run_id_raw
    else:
        m_run = re.search(r"run(\d+)", path.stem, re.IGNORECASE)
        if m_run:
            run_id = int(m_run.group(1))

    rel_parts = path.relative_to(results_dir).parts if path.is_relative_to(results_dir) else path.parts

    if not method and len(rel_parts) >= 3:
        method = rel_parts[0]
    if not model and len(rel_parts) >= 3:
        model = rel_parts[1]

    if method.lower() == "ablation" and not ablation and len(rel_parts) >= 3:
        ablation = rel_parts[1]

    if not method:
        method = "unknown"
    if not model:
        model = "unknown"

    return method, model, ablation, run_id


def _infer_mode_from_method(method: str) -> str:
    lower = method.strip().lower()
    if "full" in lower:
        return "full"
    if "base" in lower:
        return "base"
    if "baseline" in lower:
        return "base"
    return lower or "unknown"


def _split_assertions_by_summary(
    assertions: Sequence[Dict[str, object]],
    assertion_summary: Dict[str, object],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    passed: List[Dict[str, str]] = []

    failure_pool: List[Dict[str, str]] = []
    raw_failures = assertion_summary.get("failures", [])
    if isinstance(raw_failures, list):
        for item in raw_failures:
            if not isinstance(item, dict):
                continue
            failure_pool.append(
                {
                    "id": _safe_str(item.get("id")),
                    "check": _safe_str(item.get("check")),
                    "desc": _safe_str(item.get("desc")),
                    "message": _safe_str(item.get("message")),
                }
            )

    for idx, assertion in enumerate(assertions or [], start=1):
        if isinstance(assertion, dict):
            aid = _safe_str(assertion.get("id")) or f"UNKNOWN-{idx}"
            check = _safe_str(assertion.get("check") or assertion.get("condition") or assertion.get("type")) or "UNKNOWN"
            desc = _safe_str(assertion.get("desc") or assertion.get("description"))
        else:
            aid = f"UNKNOWN-{idx}"
            check = "UNKNOWN"
            desc = ""

        matched_index = -1
        for i, failure in enumerate(failure_pool):
            if failure.get("id") == aid:
                matched_index = i
                break

        if matched_index >= 0:
            # Keep the authoritative failure message from checker.
            continue

        passed.append({"id": aid, "check": check, "desc": desc})

    return passed, failure_pool


def _evaluate_single_run(
    gt: Dict[str, object],
    run: Dict[str, object],
) -> Dict[str, object]:
    graph_data = run.get("graph_snapshot")
    raw_cs = run.get("connection_set")

    G = None
    if isinstance(graph_data, dict):
        try:
            G = deserialize_graph(graph_data)
        except Exception:
            G = None

    if G is None and isinstance(raw_cs, dict):
        G = build_graph_from_connection_set(raw_cs)

    if G is None:
        raise ValueError("Result missing both valid graph_snapshot and connection_set.")

    if not isinstance(raw_cs, dict):
        raw_cs = extract_connection_set_raw(G)

    normalized_cs = normalize_connection_set(G, raw_cs)
    assertions = gt.get("verification_assertions", [])
    if not isinstance(assertions, list):
        assertions = []
    assertion_summary = evaluate_assertions(G, assertions)
    passed_assertions, failed_assertions = _split_assertions_by_summary(assertions, assertion_summary)

    pin_single_net_violations = list_multi_net_pin_violations(G)
    pin_single_net_error_count = len(pin_single_net_violations)

    benchmark_assertion_total = int(assertion_summary.get("total", 0))
    benchmark_assertion_passed = int(assertion_summary.get("passed", 0))
    adjusted_assertion_total = benchmark_assertion_total + pin_single_net_error_count
    adjusted_assertion_pass_rate = (
        benchmark_assertion_passed / adjusted_assertion_total
        if adjusted_assertion_total > 0
        else 1.0
    )

    auto_pin_single_net_failures: List[Dict[str, str]] = []
    for idx, violation in enumerate(pin_single_net_violations, start=1):
        pin = _safe_str(violation.get("pin"))
        nets = violation.get("nets", [])
        nets_text = ", ".join(_safe_str(n) for n in nets if _safe_str(n))
        auto_pin_single_net_failures.append(
            {
                "id": f"AUTO-PIN_SINGLE_NET-{idx}",
                "check": "PIN_SINGLE_NET",
                "desc": f"Auto PIN_SINGLE_NET check for {pin}",
                "message": f"Pin '{pin}' is connected to multiple nets: [{nets_text}]",
            }
        )

    all_failed_assertions = failed_assertions + auto_pin_single_net_failures

    passed = check_pass_single_with_graph(gt, G, normalized_cs.get("nets", []))
    run_meta = run.get("meta") if isinstance(run.get("meta"), dict) else {}

    return {
        "metrics": {
            "pass": passed,
            "assertion_pass_rate": _safe_float(adjusted_assertion_pass_rate, 0.0),
        },
        "assertion_details": {
            "run_id": run_meta.get("run_id"),
            "assertion_pass_rate": _safe_float(adjusted_assertion_pass_rate, 0.0),
            "benchmark_assertion_total": benchmark_assertion_total,
            "benchmark_assertion_passed": benchmark_assertion_passed,
            "pin_single_net_error_count": pin_single_net_error_count,
            "pin_single_net_violations": pin_single_net_violations,
            "assertion_total": adjusted_assertion_total,
            "assertion_failed": len(all_failed_assertions),
            "has_multi_net_pin": pin_single_net_error_count > 0,
            "passed_assertions": passed_assertions,
            "failed_assertions": all_failed_assertions,
        },
    }


def _summary(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def evaluate_design(
    gt: Dict[str, object],
    runs: Sequence[Dict[str, object]],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    n = len(runs)
    if n == 0:
        return {
            "pass_at_1": 0.0,
            "pass_at_3": 0.0,
            "assertion_pass_rate": 0.0,
        }, []

    run_metrics: List[Dict[str, object]] = []
    run_assertion_details: List[Dict[str, object]] = []
    for idx, run in enumerate(runs, start=1):
        try:
            run_eval = _evaluate_single_run(gt, run)
            metrics = run_eval.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            run_metrics.append(metrics)

            detail = run_eval.get("assertion_details", {})
            if not isinstance(detail, dict):
                detail = {}
            detail = dict(detail)
            detail["pass"] = bool(metrics.get("pass"))
            if "run_id" not in detail:
                meta = run.get("meta") if isinstance(run.get("meta"), dict) else {}
                detail["run_id"] = meta.get("run_id")
            detail["run_index"] = idx
            run_assertion_details.append(detail)
        except Exception:
            run_metrics.append(
                {
                    "pass": False,
                    "assertion_pass_rate": 0.0,
                }
            )
            run_assertion_details.append(
                {
                    "run_index": idx,
                    "run_id": None,
                    "pass": False,
                    "assertion_pass_rate": 0.0,
                    "benchmark_assertion_total": 0,
                    "benchmark_assertion_passed": 0,
                    "pin_single_net_error_count": 0,
                    "pin_single_net_violations": [],
                    "assertion_total": 0,
                    "assertion_failed": 0,
                    "has_multi_net_pin": False,
                    "passed_assertions": [],
                    "failed_assertions": [
                        {
                            "id": "RUNTIME_ERROR",
                            "check": "RUNTIME_ERROR",
                            "desc": "",
                            "message": "Run evaluation failed before assertion checking.",
                        }
                    ],
                }
            )

    assertion_rates = [_safe_float(m.get("assertion_pass_rate")) for m in run_metrics]
    assertion_mean, _ = _summary(assertion_rates)

    pass_count = sum(1 for m in run_metrics if bool(m.get("pass")))

    out: Dict[str, object] = {
        "pass_at_1": compute_pass_at_k(n, pass_count, 1),
        "pass_at_3": compute_pass_at_k(n, pass_count, 3),
        "assertion_pass_rate": assertion_mean,
        "n_runs": n,
        "pass_count": pass_count,
    }

    return out, run_assertion_details


def _build_pass_at_k_map(n: int, c: int) -> Dict[str, float]:
    if n <= 0:
        return {}
    return {
        f"pass_at_{k}": compute_pass_at_k(n, c, k)
        for k in range(1, n + 1)
    }


def _write_json_file(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _to_failed_assertion_names(failed_assertions: Sequence[Dict[str, object]]) -> List[str]:
    names: List[str] = []
    for item in failed_assertions:
        if not isinstance(item, dict):
            continue
        name = _safe_str(item.get("id")) or _safe_str(item.get("check"))
        if name:
            names.append(name)
    return names


def evaluate_all(
    assertion_dir: str = "benchmark/ground_truth_assertion",
    results_dir: str = "benchmark/results",
    output_dir: str = "benchmark/reports",
) -> List[Dict[str, object]]:
    assertion_path = _resolve_path_arg(assertion_dir)
    gt_files = _load_gt_files(assertion_path if assertion_path.exists() else None)
    if not gt_files:
        raise FileNotFoundError(
            f"No valid assertion ground truth files found under: {assertion_path}"
        )

    results_path = _resolve_path_arg(results_dir)
    cleaned = _cleanup_generated_task_summaries(results_path)
    if cleaned > 0:
        print(f"[evaluate] Removed {cleaned} stale evaluation_summary.json file(s) before this run.")

    result_paths = _discover_result_files(results_path)
    grouped_runs: Dict[Tuple[str, str, str, Optional[str]], List[Dict[str, object]]] = defaultdict(list)

    for result_path in result_paths:
        try:
            payload = _load_json(result_path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if not _is_evaluable_result_payload(payload):
            continue

        design_id = _infer_design_id(result_path, payload)
        if not design_id:
            continue
        if design_id not in gt_files:
            continue

        method, model, ablation, run_id = _infer_method_model_ablation(
            result_path,
            results_path,
            payload,
        )

        if run_id is not None:
            payload.setdefault("meta", {})
            if isinstance(payload["meta"], dict):
                payload["meta"]["run_id"] = run_id

        key = (method, model, design_id, ablation)
        grouped_runs[key].append(
            {
                "payload": payload,
                "path": str(result_path),
                "run_id": run_id,
            }
        )

    rows: List[Dict[str, object]] = []
    task_assertion_details: List[Dict[str, object]] = []
    model_task_report_rows: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)

    for (method, model, design_id, ablation), run_records in sorted(grouped_runs.items()):
        sorted_records = sorted(
            run_records,
            key=lambda x: (
                x.get("run_id") if isinstance(x.get("run_id"), int) else 10**9,
                str(x.get("path", "")),
            ),
        )
        runs = [r.get("payload", {}) for r in sorted_records if isinstance(r.get("payload", {}), dict)]
        if not runs:
            continue

        gt = gt_files[design_id]
        mode = _infer_mode_from_method(method)
        metrics, run_details = evaluate_design(gt, runs)

        pass_count = sum(1 for detail in run_details if bool(detail.get("pass")))
        n_runs = len(run_details)
        pass_at_k = _build_pass_at_k_map(n_runs, pass_count)
        avg_assertion_pass_rate = _safe_float(metrics.get("assertion_pass_rate"), 0.0)

        metrics.update(
            {
                "mode": mode,
                "method": method,
                "model": model,
                "ablation": ablation,
                "design_id": design_id,
                "tier": gt.get("tier"),
                "type": gt.get("type"),
            }
        )
        rows.append(metrics)

        task_dir = Path(str(sorted_records[0].get("path", ""))).parent
        task_run_rows: List[Dict[str, object]] = []
        for record, detail in zip(sorted_records, run_details):
            failed_assertions = detail.get("failed_assertions", [])
            if not isinstance(failed_assertions, list):
                failed_assertions = []

            task_run_rows.append(
                {
                    "run_index": detail.get("run_index"),
                    "run_id": detail.get("run_id"),
                    "result_file": Path(str(record.get("path", ""))).name,
                    "pass": bool(detail.get("pass")),
                    "assertion_pass_rate": _safe_float(detail.get("assertion_pass_rate"), 0.0),
                    "violated_assertion_count": len(failed_assertions),
                    "violated_assertion_names": _to_failed_assertion_names(failed_assertions),
                    "failed_assertions": failed_assertions,
                }
            )

        task_eval_payload: Dict[str, object] = {
            "method": method,
            "model": model,
            "ablation": ablation,
            "design_id": design_id,
            "tier": gt.get("tier"),
            "type": gt.get("type"),
            "num_runs": n_runs,
            "pass_count": pass_count,
            "pass_at_k": pass_at_k,
            "avg_assertion_pass_rate": avg_assertion_pass_rate,
            "runs": task_run_rows,
        }
        _write_json_file(task_dir / "evaluation_summary.json", task_eval_payload)

        model_task_report_rows[(method, model)].append(
            {
                "design_id": design_id,
                "tier": gt.get("tier"),
                "type": gt.get("type"),
                "ablation": ablation,
                "num_runs": n_runs,
                "pass_count": pass_count,
                "pass_at_k": pass_at_k,
                "assertion_pass_rate": avg_assertion_pass_rate,
            }
        )

        for detail in run_details:
            task_assertion_details.append(
                {
                    "mode": mode,
                    "method": method,
                    "model": model,
                    "ablation": ablation,
                    "design_id": design_id,
                    "tier": gt.get("tier"),
                    "type": gt.get("type"),
                    **detail,
                }
            )

    output_path = _resolve_path_arg(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    export_reports(rows, str(output_path))

    details_path = output_path / "assertion_task_details.json"
    with details_path.open("w", encoding="utf-8") as f:
        json.dump(task_assertion_details, f, ensure_ascii=False, indent=2)

    # Per-method/model task summary JSON under benchmark/reports/<method>/<model>/
    for (method, model), task_rows in sorted(model_task_report_rows.items()):
        sorted_tasks = sorted(task_rows, key=lambda x: _normalize_design_id_token(_safe_str(x.get("design_id"))))
        assertion_rates = [_safe_float(row.get("assertion_pass_rate"), 0.0) for row in sorted_tasks]
        pass_at_1_values = [_safe_float(row.get("pass_at_k", {}).get("pass_at_1"), 0.0) for row in sorted_tasks]
        pass_at_3_values = [_safe_float(row.get("pass_at_k", {}).get("pass_at_3"), 0.0) for row in sorted_tasks]

        model_payload: Dict[str, object] = {
            "method": method,
            "model": model,
            "task_count": len(sorted_tasks),
            "tasks": sorted_tasks,
            "overall": {
                "mean_task_pass_at_1": mean(pass_at_1_values) if pass_at_1_values else 0.0,
                "mean_task_pass_at_3": mean(pass_at_3_values) if pass_at_3_values else 0.0,
                "mean_task_assertion_pass_rate": mean(assertion_rates) if assertion_rates else 0.0,
            },
        }

        _write_json_file(output_path / method / model / "task_summary.json", model_payload)

    return rows


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate benchmark result JSON files.")
    parser.add_argument(
        "--assertion-dir",
        default="benchmark/ground_truth_assertion",
        help="Assertion ground truth directory",
    )
    parser.add_argument(
        "--results-dir",
        default="benchmark/results",
        help="Results directory or a single result JSON file",
    )
    parser.add_argument("--output-dir", default="benchmark/reports", help="Output reports directory")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    rows = evaluate_all(
        assertion_dir=args.assertion_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )
    print(f"Evaluated {len(rows)} design groups. Reports saved to {args.output_dir}.")


if __name__ == "__main__":
    main()
