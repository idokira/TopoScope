from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PASS = "pass"
FAIL = "fail"
TIER_ORDER = ("T1", "T2", "T3")


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


def merge_confusions(parts: Iterable[Confusion]) -> Confusion:
    out = Confusion()
    for c in parts:
        out.tp += c.tp
        out.fp += c.fp
        out.fn += c.fn
        out.tn += c.tn
    return out


def parse_tier(design_id_run_id: str) -> Optional[str]:
    text = design_id_run_id.strip()
    m = re.match(r"^(T[123])-[^/]+(?:/\d+)?$", text)
    if not m:
        return None
    return m.group(1)


def normalize_label(value: str) -> Optional[str]:
    v = value.strip().lower()
    if v in {PASS, FAIL}:
        return v
    return None


def verifier_label(assertion_pass_rate_text: str, threshold: float) -> Optional[str]:
    try:
        score = float(assertion_pass_rate_text)
    except ValueError:
        return None
    # Requirement: assertion_pass_rate > 90% is pass.
    return PASS if score > threshold else FAIL


def compute_cohens_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0

    po = (tp + tn) / n
    pred_p = (tp + fp) / n
    pred_f = (fn + tn) / n
    exp_p = (tp + fn) / n
    exp_f = (fp + tn) / n
    pe = pred_p * exp_p + pred_f * exp_f

    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1.0 - pe)


def compute_precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def update_confusion(c: Confusion, verifier: str, expert: str) -> None:
    if verifier == PASS and expert == PASS:
        c.tp += 1
    elif verifier == PASS and expert == FAIL:
        c.fp += 1
    elif verifier == FAIL and expert == PASS:
        c.fn += 1
    else:
        c.tn += 1


def compute_confusion(rows: Iterable[Dict[str, str]], threshold: float) -> Confusion:
    c = Confusion()
    for row in rows:
        expert = normalize_label(row.get("final_decision", ""))
        verifier = verifier_label(row.get("assertion_pass_rate", ""), threshold)
        if expert is None or verifier is None:
            continue
        update_confusion(c, verifier, expert)
    return c


def split_rows_by_tier(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {tier: [] for tier in TIER_ORDER}
    for row in rows:
        tier = parse_tier(row.get("design_id/run_id", ""))
        if tier in grouped:
            grouped[tier].append(row)
    return grouped


def _threshold_candidates(min_threshold: float, max_threshold: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("--search-step must be > 0")
    if min_threshold > max_threshold:
        raise ValueError("--search-min must be <= --search-max")

    count = int(round((max_threshold - min_threshold) / step)) + 1
    values: List[float] = []
    for i in range(count):
        t = min_threshold + i * step
        if t > max_threshold + 1e-12:
            break
        values.append(round(t, 10))

    if not values or values[-1] < max_threshold - 1e-12:
        values.append(round(max_threshold, 10))

    # Remove duplicates after rounding.
    deduped = sorted(set(values))
    return deduped


def find_best_threshold(
    rows: Iterable[Dict[str, str]],
    min_threshold: float,
    max_threshold: float,
    step: float,
) -> Tuple[float, Confusion, float, float, float, float]:
    best_threshold: Optional[float] = None
    best_confusion: Optional[Confusion] = None
    best_metrics: Optional[Tuple[float, float, float, float]] = None
    best_key: Optional[Tuple[float, float, float, float, float]] = None

    for threshold in _threshold_candidates(min_threshold, max_threshold, step):
        c = compute_confusion(rows, threshold)
        kappa = compute_cohens_kappa(c.tp, c.fp, c.fn, c.tn)
        precision, recall, f1 = compute_precision_recall_f1(c.tp, c.fp, c.fn)

        # Optimize by kappa first; tie-break by F1/precision/recall; finally prefer lower threshold.
        key = (kappa, f1, precision, recall, -threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_confusion = c
            best_metrics = (kappa, precision, recall, f1)

    if best_threshold is None or best_confusion is None or best_metrics is None:
        raise RuntimeError("No threshold candidates generated.")

    return (
        best_threshold,
        best_confusion,
        best_metrics[0],
        best_metrics[1],
        best_metrics[2],
        best_metrics[3],
    )


def _fmt_percent(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_kappa(x: float) -> str:
    return f"{x:.3f}"


def _fmt_threshold(x: float) -> str:
    return f"{x:.3f}"


def _section_title(index: int, tier_name: str, n: int, kappa: float, threshold: float) -> str:
    return (
        f"({chr(ord('a') + index)}) {tier_name} Best (threshold = {_fmt_threshold(threshold)}, "
        f"N = {n}, kappa = {_fmt_kappa(kappa)})"
    )


def render_confusion_table(c: Confusion) -> str:
    pred_p = c.tp + c.fp
    pred_f = c.fn + c.tn
    exp_p = c.tp + c.fn
    exp_f = c.fp + c.tn

    lines = [
        "|        | Expert: P | Expert: F | Total |",
        "|---|---:|---:|---:|",
        f"| Ver: P | {c.tp} (TP) | {c.fp} (FP) | {pred_p} |",
        f"| Ver: F | {c.fn} (FN) | {c.tn} (TN) | {pred_f} |",
        f"| Total | {exp_p} | {exp_f} | {c.n} |",
    ]
    return "\n".join(lines)


def render_summary(
    metrics_by_tier: Dict[str, Tuple[float, Confusion, float, float, float, float]]
) -> str:
    lines = [
        "(e) Summary of Best Threshold Search",
        "| Difficulty | Best Threshold | N | kappa | Precision | Recall | F1 Score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name in TIER_ORDER:
        threshold, c, kappa, precision, recall, f1 = metrics_by_tier[name]
        lines.append(
            f"| {name} | {_fmt_threshold(threshold)} | {c.n} | {_fmt_kappa(kappa)} | {_fmt_percent(precision)} | {_fmt_percent(recall)} | {_fmt_percent(f1)} |"
        )

    overall_threshold_text = "/".join(
        f"{name}:{_fmt_threshold(metrics_by_tier[name][0])}" for name in TIER_ORDER
    )
    overall_threshold, overall_c, overall_kappa, overall_precision, overall_recall, overall_f1 = metrics_by_tier[
        "Overall"
    ]
    _ = overall_threshold
    lines.append(
        f"| Overall | {overall_threshold_text} | {overall_c.n} | {_fmt_kappa(overall_kappa)} | {_fmt_percent(overall_precision)} | {_fmt_percent(overall_recall)} | {_fmt_percent(overall_f1)} |"
    )

    return "\n".join(lines)


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]

    # Pre-process rows to set assertion_pass_rate to 0 if there are multi-net pin violations
    import sys
    import json
    
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
    from benchmark.eval.assertion_checker import list_multi_net_pin_violations
    from benchmark.eval.graph_utils import build_graph_from_connection_set

    method_dirs = {
        "tcs_only": project_root / "benchmark/results/ablation/TCS_only",
        "erc_only": project_root / "benchmark/results/ablation/erc_only",
        "baseline": project_root / "benchmark/results/baseline",
        "full_system": project_root / "benchmark/results/full_system",
    }
    
    for row in rows:
        method = row.get("method", "")
        model = row.get("model", "")
        design_run = row.get("design_id/run_id", "")
        
        parts = design_run.split("/")
        design_id = parts[0] if len(parts) > 0 else ""
        run_id = parts[1] if len(parts) > 1 else "1"
        
        mdir = method_dirs.get(method.lower())
        if not mdir:
            continue
            
        json_path = mdir / model / design_id / f"{design_id}_run{run_id}.json"
        if not json_path.exists():
            continue
            
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            conn_set = data.get("connection_set")
            if conn_set:
                G = build_graph_from_connection_set(conn_set)
                if len(list_multi_net_pin_violations(G)) > 0:
                    row["assertion_pass_rate"] = "0.0"
                    print(f"Metrics: Multi-net pin violation found! Target set to 0. Index: {row.get('index', 'N/A')}, Design: {design_run}, Model: {model}")
        except Exception:
            pass
            
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search best threshold per T1/T2/T3 and report Cohen's Kappa / Precision / Recall / F1."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="validation_results.csv",
        help="Path to validation CSV (default: validation_results.csv)",
    )
    parser.add_argument(
        "--search-min",
        type=float,
        default=0.75,
        help="Minimum threshold in grid search (inclusive).",
    )
    parser.add_argument(
        "--search-max",
        type=float,
        default=0.95,
        help="Maximum threshold in grid search (inclusive).",
    )
    parser.add_argument(
        "--search-step",
        type=float,
        default=0.005,
        help="Threshold step size in grid search.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    rows = load_rows(csv_path)

    grouped = split_rows_by_tier(rows)
    metrics: Dict[str, Tuple[float, Confusion, float, float, float, float]] = {}

    for tier_name in TIER_ORDER:
        metrics[tier_name] = find_best_threshold(
            rows=grouped[tier_name],
            min_threshold=args.search_min,
            max_threshold=args.search_max,
            step=args.search_step,
        )

    overall_confusion = merge_confusions(metrics[name][1] for name in TIER_ORDER)
    overall_kappa = compute_cohens_kappa(
        overall_confusion.tp,
        overall_confusion.fp,
        overall_confusion.fn,
        overall_confusion.tn,
    )
    overall_precision, overall_recall, overall_f1 = compute_precision_recall_f1(
        overall_confusion.tp,
        overall_confusion.fp,
        overall_confusion.fn,
    )
    metrics["Overall"] = (
        -1.0,
        overall_confusion,
        overall_kappa,
        overall_precision,
        overall_recall,
        overall_f1,
    )

    for idx, tier_name in enumerate(TIER_ORDER):
        threshold, c, kappa, _, _, _ = metrics[tier_name]
        print(_section_title(idx, tier_name, c.n, kappa, threshold))
        print(render_confusion_table(c))
        print()

    overall_threshold_text = ", ".join(
        f"{name}={_fmt_threshold(metrics[name][0])}" for name in TIER_ORDER
    )
    overall_threshold, overall_c, overall_kappa, _, _, _ = metrics["Overall"]
    _ = overall_threshold
    print(
        f"(d) Overall Tasks (thresholds: {overall_threshold_text}, N = {overall_c.n}, "
        f"kappa = {_fmt_kappa(overall_kappa)})"
    )
    print(render_confusion_table(overall_c))
    print()

    print(render_summary(metrics))


if __name__ == "__main__":
    main()
