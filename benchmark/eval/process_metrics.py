from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple


DEFAULT_NUMERIC_KEYS: Tuple[str, ...] = (
    "erc_correction_rounds",
    "net_reuse_count",
    "net_create_count",
    "hallucination_rejections",
)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "sum": 0.0,
        }

    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "sum": float(sum(values)),
    }


def summarize_process_metrics(runs: Iterable[Dict[str, object]]) -> Dict[str, object]:
    """
    Aggregate process_metrics across multiple runs and return a flat summary.
    """
    metric_objs: List[Dict[str, object]] = []
    for run in runs:
        metrics = run.get("process_metrics")
        if isinstance(metrics, dict):
            metric_objs.append(metrics)

    out: Dict[str, object] = {
        "process_metrics_runs": len(metric_objs),
    }

    if not metric_objs:
        return out

    numeric_pool: Dict[str, List[float]] = defaultdict(list)
    modules_processed_sizes: List[float] = []
    per_module_loops: Dict[str, List[float]] = defaultdict(list)

    for metrics in metric_objs:
        for k, v in metrics.items():
            if _is_number(v):
                numeric_pool[k].append(float(v))

        modules_processed = metrics.get("modules_processed")
        if isinstance(modules_processed, list):
            modules_processed_sizes.append(float(len(modules_processed)))

        loops = metrics.get("per_module_loops")
        if isinstance(loops, dict):
            for module_name, loop_count in loops.items():
                if _is_number(loop_count):
                    per_module_loops[str(module_name)].append(float(loop_count))

    all_numeric_keys = set(DEFAULT_NUMERIC_KEYS) | set(numeric_pool.keys())
    for key in sorted(all_numeric_keys):
        stats = _summary(numeric_pool.get(key, []))
        for stat_name, stat_value in stats.items():
            out[f"pm_{key}_{stat_name}"] = stat_value

    module_size_stats = _summary(modules_processed_sizes)
    for stat_name, stat_value in module_size_stats.items():
        out[f"pm_modules_processed_count_{stat_name}"] = stat_value

    for module_name, values in sorted(per_module_loops.items()):
        stats = _summary(values)
        safe_module = module_name.replace(" ", "_")
        out[f"pm_per_module_loop_{safe_module}_mean"] = stats["mean"]
        out[f"pm_per_module_loop_{safe_module}_max"] = stats["max"]

    return out
