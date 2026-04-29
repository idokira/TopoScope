from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from benchmark.eval.assertion_checker import list_multi_net_pin_violations
    from benchmark.eval.graph_utils import build_graph_from_connection_set, deserialize_graph
except ModuleNotFoundError:
    from assertion_checker import list_multi_net_pin_violations
    from graph_utils import build_graph_from_connection_set, deserialize_graph


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TCS_ROOT = PROJECT_ROOT / "benchmark/results/ablation/TCS_only"
DEFAULT_ERC_ROOT = PROJECT_ROOT / "benchmark/results/ablation/erc_only"
DEFAULT_BASELINE_ROOT = PROJECT_ROOT / "benchmark/results/baseline"
DEFAULT_FULL_ROOT = PROJECT_ROOT / "benchmark/results/full_system"
DEFAULT_COMPONENT_DB = PROJECT_ROOT / "benchmark/component_data/Extracted_Components.json"
DEFAULT_TASK_REPORT_CSV = PROJECT_ROOT / "benchmark/reports/hallucination_task_report.csv"
DEFAULT_MODEL_REPORT_CSV = PROJECT_ROOT / "benchmark/reports/hallucination_model_report.csv"
DEFAULT_RUN_REPORT_CSV = PROJECT_ROOT / "benchmark/reports/hallucination_run_report.csv"
DEFAULT_REPORT_MD = PROJECT_ROOT / "benchmark/reports/hallucination_report.md"

METHOD_DIRS = {
    "tcs_only": DEFAULT_TCS_ROOT,
    "erc_only": DEFAULT_ERC_ROOT,
    "baseline": DEFAULT_BASELINE_ROOT,
    "full_system": DEFAULT_FULL_ROOT,
}

PASSIVE_TWO_PIN_PREFIXES = ("CAP", "RES", "IND", "FUSE", "CRYSTAL", "XTAL")
DIODE_PREFIXES = ("DIODE", "LED")

CRITICAL_NAME_KEYWORDS = (
    "EN",
    "ENABLE",
    "RESET",
    "RST",
    "NRST",
    "SHDN",
    "PDN",
    "PWDN",
    "BYP",
    "BYPASS",
    "FB",
    "VREF",
)


@dataclass
class PinSpec:
    name: str
    number: str
    electrical_type: str


@dataclass
class PartSpec:
    part_number: str
    pins: List[PinSpec]
    required_non_floating_pins: Set[str]


@dataclass
class RunDetection:
    method: str
    model: str
    task: str
    run_id: int
    class1_count: int
    class2_count: int
    class3_count: int

    @property
    def class1(self) -> int:
        return 1 if self.class1_count > 0 else 0

    @property
    def class2(self) -> int:
        return 1 if self.class2_count > 0 else 0

    @property
    def class3(self) -> int:
        return 1 if self.class3_count > 0 else 0

    @property
    def hallucinated(self) -> int:
        return 1 if (self.class1 or self.class2 or self.class3) else 0


@dataclass
class TaskAggregate:
    method: str
    model: str
    task: str
    runs: int
    class1_rate: float
    class2_rate: float
    class3_rate: float
    hallucination_rate: float


def _resolve_path(path_value: str) -> Path:
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _safe_str(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _norm_token(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", text.strip().upper())


def _parse_pin_token(raw_pin: str) -> Tuple[Optional[str], str]:
    token = _safe_str(raw_pin)
    if not token:
        return None, ""

    if "." in token:
        comp, pin = token.split(".", 1)
        return comp.strip() or None, pin.strip()

    m = re.match(r"^(?P<comp>.+)_Pin(?P<pin>[^_]+)$", token)
    if m:
        comp = m.group("comp").strip()
        pin = m.group("pin").strip()
        return comp or None, pin

    return None, token


class ComponentLibrary:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.by_part: Dict[str, Dict[str, object]] = {}
        self.by_component: Dict[str, Dict[str, object]] = {}
        self.cache: Dict[str, Optional[PartSpec]] = {}
        self._load()

    def _load(self) -> None:
        if not self.db_path.exists():
            return

        with self.db_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return

        for item in data:
            if not isinstance(item, dict):
                continue

            part = _safe_str(item.get("part_number")).lower()
            comp = _safe_str(item.get("component")).lower()

            if part:
                self.by_part[part] = item
            if comp:
                self.by_component[comp] = item

    def _to_part_spec(self, item: Dict[str, object]) -> Optional[PartSpec]:
        pins_raw = item.get("pins")
        if not isinstance(pins_raw, list):
            return None

        pins: List[PinSpec] = []
        for p in pins_raw:
            if not isinstance(p, dict):
                continue
            pins.append(
                PinSpec(
                    name=_safe_str(p.get("name")),
                    number=_safe_str(p.get("number")),
                    electrical_type=_safe_str(p.get("electrical_type")),
                )
            )

        if not pins:
            return None

        required_non_floating = self._extract_required_non_floating_pins(item)
        return PartSpec(
            part_number=_safe_str(item.get("part_number")),
            pins=pins,
            required_non_floating_pins=required_non_floating,
        )

    def _extract_required_non_floating_pins(self, item: Dict[str, object]) -> Set[str]:
        out: Set[str] = set()
        raw_rules = item.get("erc_rules")
        if not isinstance(raw_rules, list):
            return out

        include_rule_types = {
            "FORBID_FLOATING",
            "PIN_NOT_FLOATING",
            "MANDATORY_NET_CONNECTION",
        }

        for rule in raw_rules:
            if not isinstance(rule, dict):
                continue
            rule_type = _safe_str(rule.get("rule_type")).upper()
            if rule_type not in include_rule_types:
                continue

            params = rule.get("parameters")
            if not isinstance(params, dict):
                continue

            target_pins = params.get("target_pins")
            if not isinstance(target_pins, list):
                continue

            for pin in target_pins:
                pin_norm = _norm_token(_safe_str(pin))
                if pin_norm:
                    out.add(pin_norm)

        return out

    def resolve(self, part_number: str) -> Optional[PartSpec]:
        key = _safe_str(part_number).lower()
        if key in self.cache:
            return self.cache[key]

        item = self.by_part.get(key) or self.by_component.get(key)

        # Fallback contains-match for generic aliases.
        if item is None and key:
            for candidate_key, candidate_item in self.by_part.items():
                if key in candidate_key:
                    item = candidate_item
                    break

        spec = self._to_part_spec(item) if item else None
        self.cache[key] = spec
        return spec


def _passive_fallback_pins(part_number: str, refdes: str) -> Optional[PartSpec]:
    part_upper = _safe_str(part_number).upper()
    ref_upper = _safe_str(refdes).upper()

    if part_upper.startswith(PASSIVE_TWO_PIN_PREFIXES) or ref_upper.startswith(("R", "C", "L", "F", "Y")):
        return PartSpec(
            part_number=part_number,
            pins=[
                PinSpec(name="1", number="1", electrical_type="Passive"),
                PinSpec(name="2", number="2", electrical_type="Passive"),
            ],
            required_non_floating_pins=set(),
        )

    if part_upper.startswith(DIODE_PREFIXES) or ref_upper.startswith("D"):
        return PartSpec(
            part_number=part_number,
            pins=[
                PinSpec(name="A", number="1", electrical_type="Passive"),
                PinSpec(name="K", number="2", electrical_type="Passive"),
            ],
            required_non_floating_pins=set(),
        )

    return None


def _build_pin_lookup(part_spec: PartSpec) -> Dict[str, PinSpec]:
    lookup: Dict[str, PinSpec] = {}
    for pin in part_spec.pins:
        if pin.name:
            lookup[_norm_token(pin.name)] = pin
        if pin.number:
            lookup[_norm_token(pin.number)] = pin
    return lookup


def _is_passive_component(part_number: str, refdes: str) -> bool:
    part_upper = _safe_str(part_number).upper()
    ref_upper = _safe_str(refdes).upper()
    if part_upper.startswith(PASSIVE_TWO_PIN_PREFIXES):
        return True
    if part_upper.startswith(DIODE_PREFIXES):
        return True
    return ref_upper.startswith(("R", "C", "L", "D", "F", "Y"))


def _is_ground_pin(pin: PinSpec) -> bool:
    et = _safe_str(pin.electrical_type).lower()
    if "ground" in et:
        return True

    pin_name = _norm_token(pin.name)
    return pin_name in {"GND", "VSS", "AGND", "DGND", "PGND", "GROUND"}


def _is_power_pin(pin: PinSpec) -> bool:
    et = _safe_str(pin.electrical_type).lower()
    if "power" in et:
        return True

    name = _norm_token(pin.name)
    return (
        name.startswith("VCC")
        or name.startswith("VDD")
        or name.startswith("VIN")
        or name.startswith("VOUT")
        or name.startswith("VBAT")
        or name.startswith("VDDA")
        or name.startswith("VSYS")
    )


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def _extract_component_map(connection_set: Dict[str, object]) -> Dict[str, str]:
    comp_map: Dict[str, str] = {}
    raw_components = connection_set.get("components")
    if not isinstance(raw_components, list):
        return comp_map

    for comp in raw_components:
        if not isinstance(comp, dict):
            continue
        refdes = _safe_str(comp.get("refdes"))
        if not refdes:
            continue
        comp_map[refdes] = _safe_str(comp.get("part_number"))
    return comp_map


def _extract_nets(connection_set: Dict[str, object]) -> List[Dict[str, object]]:
    nets = connection_set.get("nets")
    if isinstance(nets, list):
        return [n for n in nets if isinstance(n, dict)]
    return []


def detect_hallucinations_for_run(result_path: Path, library: ComponentLibrary) -> RunDetection:
    data = _load_json(result_path)

    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    method = _safe_str(meta.get("method"))
    model = _safe_str(meta.get("model"))
    task = _safe_str(meta.get("design_id"))

    run_id_val = meta.get("run_id")
    run_id = int(run_id_val) if isinstance(run_id_val, int) else 0
    if run_id <= 0:
        m = re.search(r"_run(\d+)\.json$", result_path.name)
        if m:
            run_id = int(m.group(1))

    connection_set = data.get("connection_set") if isinstance(data.get("connection_set"), dict) else {}
    comp_map = _extract_component_map(connection_set)
    nets = _extract_nets(connection_set)

    graph_data = data.get("graph_snapshot") if isinstance(data.get("graph_snapshot"), dict) else None
    G = None
    if graph_data is not None:
        try:
            G = deserialize_graph(graph_data)
        except Exception:
            G = None
    if G is None:
        G = build_graph_from_connection_set(connection_set)

    part_specs: Dict[str, Optional[PartSpec]] = {}
    pin_lookups: Dict[str, Dict[str, PinSpec]] = {}

    for refdes, part_number in comp_map.items():
        spec = library.resolve(part_number)
        if spec is None:
            spec = _passive_fallback_pins(part_number, refdes)
        elif _is_passive_component(part_number, refdes) and not spec.required_non_floating_pins:
            # For passive parts, default to non-floating on all known pins.
            required = set()
            for pin in spec.pins:
                if pin.name:
                    required.add(_norm_token(pin.name))
                if pin.number:
                    required.add(_norm_token(pin.number))
            spec = PartSpec(
                part_number=spec.part_number,
                pins=spec.pins,
                required_non_floating_pins=required,
            )
        part_specs[refdes] = spec
        pin_lookups[refdes] = _build_pin_lookup(spec) if spec else {}

    # Build connected-pin map per component.
    connected_per_comp: Dict[str, Set[str]] = {refdes: set() for refdes in comp_map}

    class1_count = 0
    class2_count = 0
    class3_count = 0

    # Class2 is aligned to evaluator global prerequisite check: PIN_SINGLE_NET.
    class2_count = len(list_multi_net_pin_violations(G))

    for net in nets:
        raw_pins = net.get("pins")
        if not isinstance(raw_pins, list):
            continue

        for raw_pin in raw_pins:
            comp, pin_id_raw = _parse_pin_token(_safe_str(raw_pin))
            if not comp:
                continue

            pin_norm = _norm_token(pin_id_raw)
            if not pin_norm:
                continue

            if comp not in comp_map:
                class1_count += 1
                continue

            connected_per_comp.setdefault(comp, set()).add(pin_norm)

            lookup = pin_lookups.get(comp, {})
            pin_spec = lookup.get(pin_norm)
            if pin_spec is None:
                # Class 1: phantom pin reference (pin does not exist for that component).
                if part_specs.get(comp) is not None:
                    class1_count += 1
                continue

    # Class 3: floating critical pins.
    for refdes, spec in part_specs.items():
        if spec is None:
            continue

        connected = connected_per_comp.get(refdes, set())
        if not spec.required_non_floating_pins:
            continue

        for pin in spec.pins:
            pin_name_norm = _norm_token(pin.name)
            pin_num_norm = _norm_token(pin.number)
            if pin_name_norm not in spec.required_non_floating_pins and pin_num_norm not in spec.required_non_floating_pins:
                continue

            candidates = set()
            if pin.name:
                candidates.add(_norm_token(pin.name))
            if pin.number:
                candidates.add(_norm_token(pin.number))

            if not candidates.intersection(connected):
                class3_count += 1

    return RunDetection(
        method=method,
        model=model,
        task=task,
        run_id=run_id,
        class1_count=class1_count,
        class2_count=class2_count,
        class3_count=class3_count,
    )


def _list_model_task_pairs(tcs_root: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not tcs_root.exists():
        return pairs

    for model_dir in sorted([p for p in tcs_root.iterdir() if p.is_dir()]):
        for task_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            task = task_dir.name
            if re.match(r"^T[123]-\d+$", task):
                pairs.append((model_dir.name, task))
    return pairs


def _collect_run_files(method_root: Path, model: str, task: str) -> List[Path]:
    task_dir = method_root / model / task
    if not task_dir.exists() or not task_dir.is_dir():
        return []

    files = sorted(task_dir.glob(f"{task}_run*.json"))
    return [p for p in files if p.is_file()]


def _aggregate_task(method: str, model: str, task: str, runs: Sequence[RunDetection]) -> TaskAggregate:
    n = len(runs)
    if n == 0:
        return TaskAggregate(
            method=method,
            model=model,
            task=task,
            runs=0,
            class1_rate=0.0,
            class2_rate=0.0,
            class3_rate=0.0,
            hallucination_rate=0.0,
        )

    class1 = sum(r.class1 for r in runs)
    class2 = sum(r.class2 for r in runs)
    class3 = sum(r.class3 for r in runs)
    any_h = sum(r.hallucinated for r in runs)

    return TaskAggregate(
        method=method,
        model=model,
        task=task,
        runs=n,
        class1_rate=class1 / n,
        class2_rate=class2 / n,
        class3_rate=class3 / n,
        hallucination_rate=any_h / n,
    )


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def write_run_report_csv(path: Path, rows: Sequence[RunDetection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "model",
                "task",
                "run_id",
                "class1_phantom_pin",
                "class2_role_abuse",
                "class3_floating_critical",
                "hallucinated_any",
                "class1_count",
                "class2_count",
                "class3_count",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.method,
                    r.model,
                    r.task,
                    r.run_id,
                    r.class1,
                    r.class2,
                    r.class3,
                    r.hallucinated,
                    r.class1_count,
                    r.class2_count,
                    r.class3_count,
                ]
            )


def write_task_report_csv(
    path: Path,
    model_task_pairs: Sequence[Tuple[str, str]],
    task_stats: Dict[Tuple[str, str, str], TaskAggregate],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "task",
                "tcs_runs",
                "tcs_hallucination_rate",
                "tcs_class1_rate",
                "tcs_class2_rate",
                "tcs_class3_rate",
                "erc_runs",
                "erc_hallucination_rate",
                "erc_class1_rate",
                "erc_class2_rate",
                "erc_class3_rate",
                "baseline_runs",
                "baseline_hallucination_rate",
                "baseline_class1_rate",
                "baseline_class2_rate",
                "baseline_class3_rate",
                "full_runs",
                "full_hallucination_rate",
                "full_class1_rate",
                "full_class2_rate",
                "full_class3_rate",
            ]
        )

        for model, task in model_task_pairs:
            tcs = task_stats[("tcs_only", model, task)]
            erc = task_stats[("erc_only", model, task)]
            base = task_stats[("baseline", model, task)]
            full = task_stats[("full_system", model, task)]

            writer.writerow(
                [
                    model,
                    task,
                    tcs.runs,
                    round(tcs.hallucination_rate, 6),
                    round(tcs.class1_rate, 6),
                    round(tcs.class2_rate, 6),
                    round(tcs.class3_rate, 6),
                    erc.runs,
                    round(erc.hallucination_rate, 6),
                    round(erc.class1_rate, 6),
                    round(erc.class2_rate, 6),
                    round(erc.class3_rate, 6),
                    base.runs,
                    round(base.hallucination_rate, 6),
                    round(base.class1_rate, 6),
                    round(base.class2_rate, 6),
                    round(base.class3_rate, 6),
                    full.runs,
                    round(full.hallucination_rate, 6),
                    round(full.class1_rate, 6),
                    round(full.class2_rate, 6),
                    round(full.class3_rate, 6),
                ]
            )


def write_model_report_csv(
    path: Path,
    model_task_pairs: Sequence[Tuple[str, str]],
    task_stats: Dict[Tuple[str, str, str], TaskAggregate],
) -> None:
    model_to_tasks: Dict[str, List[str]] = {}
    for model, task in model_task_pairs:
        model_to_tasks.setdefault(model, []).append(task)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "task_count",
                "tcs_avg_hallucination_rate",
                "erc_avg_hallucination_rate",
                "baseline_avg_hallucination_rate",
                "full_avg_hallucination_rate",
            ]
        )

        for model in sorted(model_to_tasks):
            tasks = sorted(set(model_to_tasks[model]))
            n = len(tasks)
            if n == 0:
                continue

            tcs_avg = sum(task_stats[("tcs_only", model, t)].hallucination_rate for t in tasks) / n
            erc_avg = sum(task_stats[("erc_only", model, t)].hallucination_rate for t in tasks) / n
            base_avg = sum(task_stats[("baseline", model, t)].hallucination_rate for t in tasks) / n
            full_avg = sum(task_stats[("full_system", model, t)].hallucination_rate for t in tasks) / n

            writer.writerow([model, n, round(tcs_avg, 6), round(erc_avg, 6), round(base_avg, 6), round(full_avg, 6)])


def write_markdown_report(
    path: Path,
    model_task_pairs: Sequence[Tuple[str, str]],
    task_stats: Dict[Tuple[str, str, str], TaskAggregate],
) -> None:
    model_to_tasks: Dict[str, List[str]] = {}
    for model, task in model_task_pairs:
        model_to_tasks.setdefault(model, []).append(task)

    lines: List[str] = []
    lines.append("# Hallucination Rate Report: TCS_only vs ERC_only vs Baseline vs Full")
    lines.append("")
    lines.append("Definitions:")
    lines.append("- Class 1: Phantom pin reference (pin does not exist on the referenced component).")
    lines.append("- Class 2: PIN_SINGLE_NET violations only (same as evaluator global pre-check for multi-net pin).")
    lines.append("- Class 3: Floating non-floating-required pins: erc_rules target_pins plus all passive component pins (R/C/L/D/Y/F) by default.")
    lines.append("- Hallucination rate: fraction of runs with any class triggered.")
    lines.append("")

    lines.append("## Model-Level Average Hallucination Rate")
    lines.append("")
    lines.append("| Model | Task Count | TCS_only Avg | ERC_only Avg | Baseline Avg | Full Avg |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for model in sorted(model_to_tasks):
        tasks = sorted(set(model_to_tasks[model]))
        n = len(tasks)
        if n == 0:
            continue

        tcs_avg = sum(task_stats[("tcs_only", model, t)].hallucination_rate for t in tasks) / n
        erc_avg = sum(task_stats[("erc_only", model, t)].hallucination_rate for t in tasks) / n
        base_avg = sum(task_stats[("baseline", model, t)].hallucination_rate for t in tasks) / n
        full_avg = sum(task_stats[("full_system", model, t)].hallucination_rate for t in tasks) / n

        lines.append(
            f"| {model} | {n} | {_fmt_pct(tcs_avg)} | {_fmt_pct(erc_avg)} | {_fmt_pct(base_avg)} | {_fmt_pct(full_avg)} |"
        )

    lines.append("")
    lines.append("## Per-Task Average Hallucination Rate (7 runs target)")
    lines.append("")

    for model in sorted(model_to_tasks):
        lines.append(f"### {model}")
        lines.append("")
        lines.append(
            "| Task | TCS runs | TCS Halluc. | ERC runs | ERC Halluc. | Baseline runs | Baseline Halluc. | Full runs | Full Halluc. |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

        for task in sorted(set(model_to_tasks[model])):
            tcs = task_stats[("tcs_only", model, task)]
            erc = task_stats[("erc_only", model, task)]
            base = task_stats[("baseline", model, task)]
            full = task_stats[("full_system", model, task)]

            lines.append(
                "| "
                f"{task} | {tcs.runs} | {_fmt_pct(tcs.hallucination_rate)} | "
                f"{erc.runs} | {_fmt_pct(erc.hallucination_rate)} | "
                f"{base.runs} | {_fmt_pct(base.hallucination_rate)} | "
                f"{full.runs} | {_fmt_pct(full.hallucination_rate)} |"
            )

        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect hallucination classes in TCS_only and compare with erc_only/baseline/full per model-task."
    )
    parser.add_argument("--tcs-root", default=str(DEFAULT_TCS_ROOT))
    parser.add_argument("--erc-root", default=str(DEFAULT_ERC_ROOT))
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--full-root", default=str(DEFAULT_FULL_ROOT))
    parser.add_argument("--component-db", default=str(DEFAULT_COMPONENT_DB))
    parser.add_argument("--task-report-csv", default=str(DEFAULT_TASK_REPORT_CSV))
    parser.add_argument("--model-report-csv", default=str(DEFAULT_MODEL_REPORT_CSV))
    parser.add_argument("--run-report-csv", default=str(DEFAULT_RUN_REPORT_CSV))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    method_roots = {
        "tcs_only": _resolve_path(args.tcs_root),
        "erc_only": _resolve_path(args.erc_root),
        "baseline": _resolve_path(args.baseline_root),
        "full_system": _resolve_path(args.full_root),
    }
    component_db = _resolve_path(args.component_db)

    library = ComponentLibrary(component_db)
    model_task_pairs = _list_model_task_pairs(method_roots["tcs_only"])
    if not model_task_pairs:
        raise RuntimeError(f"No TCS_only model-task pairs found under: {method_roots['tcs_only']}")

    run_rows: List[RunDetection] = []
    task_stats: Dict[Tuple[str, str, str], TaskAggregate] = {}

    for method, root in method_roots.items():
        for model, task in model_task_pairs:
            run_files = _collect_run_files(root, model, task)
            detections = [detect_hallucinations_for_run(path, library) for path in run_files]

            for d in detections:
                if not d.method:
                    d.method = method
                if not d.model:
                    d.model = model
                if not d.task:
                    d.task = task

            run_rows.extend(detections)
            task_stats[(method, model, task)] = _aggregate_task(method, model, task, detections)

    run_rows.sort(key=lambda x: (x.method, x.model, x.task, x.run_id))

    run_report_csv = _resolve_path(args.run_report_csv)
    task_report_csv = _resolve_path(args.task_report_csv)
    model_report_csv = _resolve_path(args.model_report_csv)
    report_md = _resolve_path(args.report_md)

    write_run_report_csv(run_report_csv, run_rows)
    write_task_report_csv(task_report_csv, model_task_pairs, task_stats)
    write_model_report_csv(model_report_csv, model_task_pairs, task_stats)
    write_markdown_report(report_md, model_task_pairs, task_stats)

    print(f"[hallucination] model_task_pairs={len(model_task_pairs)}")
    print(f"[hallucination] run_rows={len(run_rows)}")
    print(f"[hallucination] run_report_csv={run_report_csv}")
    print(f"[hallucination] task_report_csv={task_report_csv}")
    print(f"[hallucination] model_report_csv={model_report_csv}")
    print(f"[hallucination] report_md={report_md}")


if __name__ == "__main__":
    main()
