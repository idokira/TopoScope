from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Allow running as a script: python benchmark/runners/run_all_task_full.py
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full-system pipeline for all benchmark tasks with repeated trials."
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
    parser.add_argument("--repeats", type=int, default=7, help="Number of runs per task (default: 7)")
    parser.add_argument(
        "--design-ids",
        nargs="*",
        default=None,
        help="Optional subset of design ids to run (e.g. T1-1 T2-4)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when any run fails",
    )
    return parser


def _filter_cases(cases: List[Dict[str, Any]], design_ids: List[str] | None) -> List[Dict[str, Any]]:
    if not design_ids:
        return cases

    selected = {d.strip() for d in design_ids if d and d.strip()}
    if not selected:
        return cases

    out: List[Dict[str, Any]] = []
    for case in cases:
        design_id = str(case.get("design_id", "")).strip()
        if design_id in selected:
            out.append(case)

    missing = sorted(selected - {str(c.get("design_id", "")).strip() for c in out})
    if missing:
        raise KeyError(f"Design ids not found in cases: {missing}")

    return out


def _safe_path_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())


def _existing_run_ids(results_root: Path, model_name: str, design_id: str) -> List[int]:
    model_dir = results_root / _safe_path_token(model_name)
    design_dir = model_dir / _safe_path_token(design_id)
    if not design_dir.exists():
        return []

    pattern = re.compile(rf"^{re.escape(design_id)}_run(\d+)\.json$")
    run_ids: List[int] = []
    for fp in design_dir.glob(f"{design_id}_run*.json"):
        match = pattern.match(fp.name)
        if not match:
            continue
        run_ids.append(int(match.group(1)))

    return sorted(set(run_ids))


def main() -> None:
    args = _build_parser().parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    config_path = _resolve_path_arg(args.config)
    cases_path = _resolve_path_arg(args.cases)
    results_root = _resolve_path_arg(args.results_root)

    provider_name = args.provider_override or args.provider or "qwen"
    provider_config = load_provider_config(provider=provider_name, config_path=str(config_path))
    provider = provider_config.provider
    model_name = provider_config.model

    cases = _load_cases(cases_path)
    selected_cases = _filter_cases(cases, args.design_ids)

    run_plan: Dict[str, List[int]] = {}
    done_counts: Dict[str, int] = {}
    already_completed_runs = 0

    for case in selected_cases:
        design_id = str(case.get("design_id", "")).strip()
        existing_ids = _existing_run_ids(results_root, model_name, design_id)
        existing_target_ids = {rid for rid in existing_ids if 1 <= rid <= args.repeats}

        done_counts[design_id] = len(existing_target_ids)
        already_completed_runs += len(existing_target_ids)

        run_plan[design_id] = [
            rid for rid in range(1, args.repeats + 1)
            if rid not in existing_target_ids
        ]

    scheduled_runs = sum(len(v) for v in run_plan.values())

    print("=" * 72)
    print("[RUN-ALL] full_system repeated benchmark")
    print(json.dumps(
        {
            "provider": provider,
            "model": model_name,
            "cases_file": str(cases_path),
            "results_root": str(results_root),
            "task_count": len(selected_cases),
            "repeats_per_task": args.repeats,
            "already_completed_runs": already_completed_runs,
            "scheduled_runs": scheduled_runs,
        },
        ensure_ascii=False,
        indent=2,
    ))
    print("=" * 72)

    if scheduled_runs == 0:
        print("[RUN-ALL] all requested runs are already completed for this model. Nothing to do.")
        return

    failures: List[Dict[str, Any]] = []
    total = scheduled_runs
    done = 0

    for case in selected_cases:
        design_id = str(case.get("design_id", "")).strip()
        planned_run_ids = run_plan.get(design_id, [])
        if not planned_run_ids:
            print(f"[RUN-ALL] skip design={design_id} (already done {done_counts.get(design_id, 0)}/{args.repeats})")
            continue

        for run_id in planned_run_ids:
            done += 1
            print("\n" + "-" * 72)
            print(
                f"[RUN-ALL] progress {done}/{total} | design={design_id} | "
                f"run_id={run_id}/{args.repeats}"
            )
            print("-" * 72)
            try:
                out_file = run_single_case(
                    provider=provider,
                    config_file=config_path,
                    case=case,
                    run_id=run_id,
                    results_root=results_root,
                )
                print(f"[RUN-ALL] success -> {out_file}")
            except Exception as exc:
                failure = {
                    "design_id": design_id,
                    "run_id": run_id,
                    "error": str(exc),
                }
                failures.append(failure)
                print(f"[RUN-ALL] failed -> {failure}")
                traceback.print_exc()
                if args.fail_fast:
                    raise

    print("\n" + "=" * 72)
    print("[RUN-ALL] summary")
    print(json.dumps(
        {
            "total_target_runs": len(selected_cases) * args.repeats,
            "already_completed_runs": already_completed_runs,
            "scheduled_runs": total,
            "failed_runs": len(failures),
            "succeeded_runs": total - len(failures),
        },
        ensure_ascii=False,
        indent=2,
    ))

    if failures:
        print("[RUN-ALL] failed details:")
        print(json.dumps(failures, ensure_ascii=False, indent=2))
        print("=" * 72)
        raise SystemExit(1)

    print("[RUN-ALL] all runs completed successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()
