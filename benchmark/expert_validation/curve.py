import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
fm.fontManager.addfont("./times.ttf")

# Set font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

PASS = "pass"
FAIL = "fail"
TIER_ORDER = ("T1", "T2", "T3")

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

def compute_confusion(rows: Iterable[Dict[str, str]], threshold: float):
    tp = fp = fn = tn = 0
    for row in rows:
        expert = normalize_label(row.get("final_decision", ""))
        verifier = verifier_label(row.get("assertion_pass_rate", ""), threshold)
        if expert is None or verifier is None:
            continue
        if verifier == PASS and expert == PASS:
            tp += 1
        elif verifier == PASS and expert == FAIL:
            fp += 1
        elif verifier == FAIL and expert == PASS:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn

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
                    print(f"ROC.py Multi-net pin violation found! Target set to 0. Index: {row.get('index', 'N/A')}, Design: {design_run}, Model: {model}")
        except Exception:
            pass
            
    return rows

def split_rows_by_tier(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {tier: [] for tier in TIER_ORDER}
    for row in rows:
        tier = parse_tier(row.get("design_id/run_id", ""))
        if tier in grouped:
            grouped[tier].append(row)
    return grouped

def main():
    parser = argparse.ArgumentParser(description="Plot Kappa vs Threshold")
    parser.add_argument("csv_path", nargs="?", default="validation_results.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    rows = load_rows(csv_path)
    grouped = split_rows_by_tier(rows)

    # step 0.005 equals 0.5% point 
    thresholds = np.arange(0.8, 1.001, 0.005)
    
    tier_kappas = {tier: [] for tier in TIER_ORDER}

    for t in thresholds:
        for tier in TIER_ORDER:
            tp, fp, fn, tn = compute_confusion(grouped[tier], t)
            kappa = compute_cohens_kappa(tp, fp, fn, tn)
            tier_kappas[tier].append(kappa)

    # Plot
    plt.figure(figsize=(8, 6))
    
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    import matplotlib.font_manager as fm
    font_props = fm.FontProperties(family='Times New Roman', size=12)
    legend_props = fm.FontProperties(family='Times New Roman', size=10)
    tick_props = fm.FontProperties(family='Times New Roman', size=10)

    tier_names = {"T1": "Easy", "T2": "Medium", "T3": "Hard"}

    for i, tier in enumerate(TIER_ORDER):
        plt.plot(
            thresholds, 
            tier_kappas[tier], 
            marker=markers[i],
            color=colors[i],
            linewidth=2,
            markersize=6,
            label=f'{tier_names[tier]}'
        )

    plt.xlabel('Threshold', fontproperties=font_props)
    plt.ylabel('Cohen\'s Kappa', fontproperties=font_props)
    
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(tick_props)

    plt.xlim(0.79, 1.01)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', prop=legend_props)
    
    plt.tight_layout()
    out_pdf = "kappa_vs_threshold.pdf"
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    print(f"Plot saved successfully to {out_pdf}")

if __name__ == "__main__":
    main()
