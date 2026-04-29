import sys
from pathlib import Path
import json

PROJECT_ROOT = Path("/root/autodl-tmp")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from benchmark.eval.assertion_checker import list_multi_net_pin_violations
    from benchmark.eval.graph_utils import build_graph_from_connection_set
except Exception as e:
    print(e)

METHOD_DIRS = {
    "tcs_only": PROJECT_ROOT / "benchmark/results/ablation/TCS_only",
    "erc_only": PROJECT_ROOT / "benchmark/results/ablation/erc_only",
    "baseline": PROJECT_ROOT / "benchmark/results/baseline",
    "full_system": PROJECT_ROOT / "benchmark/results/full_system",
}

def has_multi_net_error(method: str, model: str, design_run: str) -> bool:
    try:
        parts = design_run.split("/")
        design_id = parts[0]
        run_id = parts[1] if len(parts) > 1 else "1"
        
        mdir = METHOD_DIRS.get(method.lower(), None)
        if not mdir:
            return False
            
        json_path = mdir / model / design_id / f"{design_id}_run{run_id}.json"
        
        if not json_path.exists():
            return False
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        conn_set = data.get("connection_set")
        if not conn_set:
            return False
            
        G = build_graph_from_connection_set(conn_set)
        violations = list_multi_net_pin_violations(G)
        return len(violations) > 0
    except Exception as e:
        return False

# Test the function with the user's example: 217 tcs_only qwen3.5-plus T2-6/3
print("Has error?", has_multi_net_error("tcs_only", "qwen3.5-plus", "T2-6/3"))
