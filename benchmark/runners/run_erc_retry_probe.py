from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow running as a script: python benchmark/runners/run_erc_retry_probe.py
if __package__ in {None, ""}:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.runners.llm_clients import load_provider_config
from benchmark.runners.run_full_system import (
    DEFAULT_CASES_FILE,
    DEFAULT_CONFIG_FILE,
    DEFAULT_RESULTS_ROOT,
    _load_cases,
    _resolve_path_arg,
    run_single_case,
)


DEFAULT_BUNDLE_ROOT = PROJECT_ROOT / "benchmark/reports/erc_retry_probe"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one round of benchmark tasks and capture full logs for tasks whose retry count reaches threshold."
        )
    )
    parser.add_argument(
        "provider",
        nargs="?",
        default="qwen",
        help="Provider key in llm_clients config (e.g. qwen). Default: qwen",
    )
    parser.add_argument(
        "--provider",
        dest="provider_override",
        default=None,
        help="Optional override for provider key (takes precedence over positional provider).",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_FILE), help="Path to llm client config JSON")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_FILE), help="Path to benchmark cases JSON")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT), help="Root output directory")
    parser.add_argument(
        "--bundle-root",
        default=str(DEFAULT_BUNDLE_ROOT),
        help="Directory to store Tx-x_erc_error bundles and summaries",
    )
    parser.add_argument(
        "--retry-threshold",
        type=int,
        default=6,
        help="Capture logs when any component retry reaches this threshold (default: 6)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=20,
        help="How many tasks to run in sequence (default: 20)",
    )
    parser.add_argument(
        "--design-ids",
        nargs="*",
        default=None,
        help="Optional subset of design ids (e.g. T1-1 T2-3). If omitted, use case order.",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Optional fixed run_id passed to run_single_case. Default: auto-increment per task.",
    )
    return parser


def _filter_cases(cases: List[Dict[str, Any]], design_ids: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not design_ids:
        return cases

    selected = [d.strip() for d in design_ids if d and d.strip()]
    selected_set = set(selected)
    if not selected_set:
        return cases

    by_id = {str(c.get("design_id", "")).strip(): c for c in cases}
    missing = [d for d in selected if d not in by_id]
    if missing:
        raise KeyError(f"Design ids not found in cases: {missing}")

    return [by_id[d] for d in selected]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _to_int(value: Any) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _parse_run_id_from_result_path(design_id: str, result_path: Optional[Path]) -> str:
    if result_path is None:
        return "NA"
    m = re.search(rf"{re.escape(design_id)}_run(\d+)\.json$", result_path.name)
    if not m:
        return "NA"
    return m.group(1)


def _detect_retry_threshold_hit(
    design_id: str,
    result_path: Optional[Path],
    combined_log: str,
    retry_threshold: int,
) -> Tuple[bool, List[str], int, List[str], Dict[str, int]]:
    hit_reasons: List[str] = []
    hit_components: List[str] = []
    max_retry_seen = 0
    retry_map: Dict[str, int] = {}

    if result_path and result_path.exists():
        data = _read_json(result_path)
        process_metrics = data.get("process_metrics", {})
        per_component = process_metrics.get("per_component_retries", {})
        if isinstance(per_component, dict):
            for comp, raw in per_component.items():
                rv = _to_int(raw)
                if rv is None:
                    continue
                retry_map[str(comp)] = rv
                max_retry_seen = max(max_retry_seen, rv)
                if rv >= retry_threshold:
                    hit_components.append(str(comp))

        if hit_components:
            hit_reasons.append("process_metrics.per_component_retries")

    retry_regex = re.compile(
        rf"(?:重试|retry)\s*{retry_threshold}\s*/\s*{retry_threshold}",
        flags=re.IGNORECASE,
    )
    if retry_regex.search(combined_log):
        hit_reasons.append("stdout_or_stderr_retry_marker")
        max_retry_seen = max(max_retry_seen, retry_threshold)

    hit = bool(hit_reasons)

    if not hit and retry_map:
        max_retry_seen = max(max_retry_seen, max(retry_map.values()))

    if not hit and design_id and f"{retry_threshold}/{retry_threshold}" in combined_log:
        hit = True
        hit_reasons.append("generic_ratio_marker")
        max_retry_seen = max(max_retry_seen, retry_threshold)

    return hit, sorted(set(hit_components)), max_retry_seen, hit_reasons, retry_map


def _extract_erc_lines(text: str, limit: int = 400) -> str:
    lines = text.splitlines()
    selected: List[str] = []
    for line in lines:
        low = line.lower()
        if (
            "phase1 retry" in low
            or "hard_erc" in low
            or "soft_erc" in low
            or "merge_errors" in low
            or "retained soft_erc_issues" in low
            or "last hard_erc_issues" in low
            or "last merge_errors" in low
        ):
            selected.append(line)
            if len(selected) >= limit:
                break
    return "\n".join(selected)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _save_erc_error_bundle(
    bundle_root: Path,
    design_id: str,
    run_label: str,
    provider: str,
    model: str,
    case_data: Dict[str, Any],
    result_path: Optional[Path],
    stdout_text: str,
    stderr_text: str,
    elapsed_seconds: float,
    retry_threshold: int,
    hit_components: List[str],
    max_retry_seen: int,
    hit_reasons: List[str],
    retry_map: Dict[str, int],
    failed: bool,
) -> Path:
    target_dir = bundle_root / f"{design_id}_erc_error"
    target_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{design_id}_run{run_label}"
    stdout_path = target_dir / f"{prefix}_stdout.log"
    stderr_path = target_dir / f"{prefix}_stderr.log"
    excerpt_path = target_dir / f"{prefix}_erc_excerpt.log"
    meta_path = target_dir / f"{prefix}_meta.json"

    _write_text(stdout_path, stdout_text)
    _write_text(stderr_path, stderr_text)
    _write_text(excerpt_path, _extract_erc_lines(stdout_text + "\n" + stderr_text))

    copied_result: Optional[Path] = None
    if result_path and result_path.exists():
        copied_result = target_dir / f"{prefix}_result.json"
        shutil.copy2(result_path, copied_result)

    meta: Dict[str, Any] = {
        "design_id": design_id,
        "provider": provider,
        "model": model,
        "retry_threshold": retry_threshold,
        "max_retry_seen": max_retry_seen,
        "hit_components": hit_components,
        "hit_reasons": hit_reasons,
        "failed": failed,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "case": case_data,
        "result_path": str(result_path) if result_path else None,
        "copied_result_path": str(copied_result) if copied_result else None,
        "per_component_retries": retry_map,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))

    return target_dir


def main() -> None:
    args = _build_parser().parse_args()

    if args.max_tasks < 1:
        raise ValueError("--max-tasks must be >= 1")
    if args.retry_threshold < 1:
        raise ValueError("--retry-threshold must be >= 1")

    provider_name = args.provider_override or args.provider or "qwen"

    config_path = _resolve_path_arg(args.config)
    cases_path = _resolve_path_arg(args.cases)
    results_root = _resolve_path_arg(args.results_root)
    bundle_root = _resolve_path_arg(args.bundle_root)

    cfg = load_provider_config(provider=provider_name, config_path=str(config_path))

    cases = _load_cases(cases_path)
    selected_cases = _filter_cases(cases, args.design_ids)
    selected_cases = selected_cases[: args.max_tasks]

    bundle_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("[ERC-PROBE] start")
    print(
        json.dumps(
            {
                "provider": cfg.provider,
                "model": cfg.model,
                "cases_file": str(cases_path),
                "results_root": str(results_root),
                "bundle_root": str(bundle_root),
                "retry_threshold": args.retry_threshold,
                "task_count": len(selected_cases),
                "run_id": args.run_id,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=" * 72)

    summaries: List[Dict[str, Any]] = []
    hit_count = 0

    for idx, case in enumerate(selected_cases, start=1):
        design_id = str(case.get("design_id", "")).strip()
        print("\n" + "-" * 72)
        print(f"[ERC-PROBE] {idx}/{len(selected_cases)} design={design_id}")
        print("-" * 72)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        result_path: Optional[Path] = None
        failed = False

        t0 = time.time()
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            try:
                result_path = run_single_case(
                    provider=cfg.provider,
                    config_file=config_path,
                    case=case,
                    run_id=args.run_id,
                    results_root=results_root,
                )
            except Exception:
                failed = True
                traceback.print_exc()

        elapsed = time.time() - t0
        stdout_text = stdout_buf.getvalue()
        stderr_text = stderr_buf.getvalue()
        combined_log = stdout_text + "\n" + stderr_text

        hit, hit_components, max_retry_seen, hit_reasons, retry_map = _detect_retry_threshold_hit(
            design_id=design_id,
            result_path=result_path,
            combined_log=combined_log,
            retry_threshold=args.retry_threshold,
        )

        run_label = _parse_run_id_from_result_path(design_id, result_path)
        bundle_dir: Optional[Path] = None
        if hit:
            bundle_dir = _save_erc_error_bundle(
                bundle_root=bundle_root,
                design_id=design_id,
                run_label=run_label,
                provider=cfg.provider,
                model=cfg.model,
                case_data=case,
                result_path=result_path,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                elapsed_seconds=elapsed,
                retry_threshold=args.retry_threshold,
                hit_components=hit_components,
                max_retry_seen=max_retry_seen,
                hit_reasons=hit_reasons,
                retry_map=retry_map,
                failed=failed,
            )
            hit_count += 1

        summary_item: Dict[str, Any] = {
            "design_id": design_id,
            "failed": failed,
            "elapsed_seconds": round(elapsed, 3),
            "result_path": str(result_path) if result_path else None,
            "run_label": run_label,
            "retry_hit": hit,
            "max_retry_seen": max_retry_seen,
            "hit_components": hit_components,
            "hit_reasons": hit_reasons,
            "bundle_dir": str(bundle_dir) if bundle_dir else None,
            "per_component_retries": retry_map,
        }
        summaries.append(summary_item)

        status = "HIT" if hit else "OK"
        fail_text = "FAILED" if failed else "DONE"
        print(
            f"[ERC-PROBE] {design_id} {fail_text} {status} "
            f"max_retry={max_retry_seen} bundle={summary_item['bundle_dir']}"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = bundle_root / f"erc_retry_probe_summary_{cfg.provider}_{ts}.json"
    hit_only_path = bundle_root / f"erc_retry_probe_hits_{cfg.provider}_{ts}.json"

    final_summary = {
        "provider": cfg.provider,
        "model": cfg.model,
        "retry_threshold": args.retry_threshold,
        "task_count": len(selected_cases),
        "hit_count": hit_count,
        "failed_count": sum(1 for s in summaries if s["failed"]),
        "items": summaries,
    }

    _write_text(summary_path, json.dumps(final_summary, ensure_ascii=False, indent=2))
    _write_text(
        hit_only_path,
        json.dumps([s for s in summaries if s["retry_hit"]], ensure_ascii=False, indent=2),
    )

    print("\n" + "=" * 72)
    print("[ERC-PROBE] done")
    print(
        json.dumps(
            {
                "summary_file": str(summary_path),
                "hit_file": str(hit_only_path),
                "hit_count": hit_count,
                "task_count": len(selected_cases),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
