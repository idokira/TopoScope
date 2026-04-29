from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "benchmark/results"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmark/expert_validation"
DEFAULT_SAMPLE_DIR = DEFAULT_OUTPUT_ROOT / "validation_sample"
DEFAULT_CSV_PATH = DEFAULT_OUTPUT_ROOT / "validation_results.csv"

TOTAL_SAMPLES = 300
SEED = 2026

SOURCE_DIR_CANDIDATES = ("ablation", "baseline", "full", "full_system")

# (label, lower, upper, ratio)
BIN_SPECS: Sequence[Tuple[str, float, float, float]] = (
	("0-50", 0.0, 0.50, 0.10),
	("50-70", 0.50, 0.70, 0.10),
	("70-85", 0.70, 0.85, 0.20),
	("85-90", 0.85, 0.90, 0.25),
	("90-95", 0.90, 0.95, 0.25),
	("95-100", 0.95, 1.0000001, 0.10),
)


@dataclass(frozen=True)
class RunCandidate:
	result_path: Path
	method: str
	model: str
	design_id: str
	run_id: Optional[int]
	assertion_pass_rate: float


def _resolve_path_arg(path_value: str) -> Path:
	p = Path(path_value).expanduser()
	if p.is_absolute():
		return p
	cwd_candidate = (Path.cwd() / p).resolve()
	if cwd_candidate.exists():
		return cwd_candidate
	return (PROJECT_ROOT / p).resolve()


def _safe_float(value: object) -> Optional[float]:
	if isinstance(value, (int, float)):
		return float(value)
	if isinstance(value, str):
		text = value.strip().replace("%", "")
		if not text:
			return None
		try:
			num = float(text)
		except ValueError:
			return None
		if num > 1.0:
			num = num / 100.0
		return num
	return None


def _discover_summary_files(results_root: Path) -> List[Path]:
	summary_files: List[Path] = []
	seen: set[Path] = set()

	for source_dir in SOURCE_DIR_CANDIDATES:
		sub_root = results_root / source_dir
		if not sub_root.exists() or not sub_root.is_dir():
			continue
		for summary_path in sorted(sub_root.rglob("evaluation_summary.json")):
			resolved = summary_path.resolve()
			if resolved in seen:
				continue
			seen.add(resolved)
			summary_files.append(summary_path)

	return summary_files


def _design_id_run_id(candidate: RunCandidate) -> str:
	if candidate.run_id is None:
		return candidate.design_id
	return f"{candidate.design_id}/{candidate.run_id}"

def _update_existing_csv_assertion_pass_rate(
	csv_path: Path,
	candidates_for_update: Sequence[RunCandidate],
) -> None:
	with csv_path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		fieldnames = reader.fieldnames
		if not fieldnames:
			raise RuntimeError(f"CSV has no header: {csv_path}")
		rows = list(reader)

	if "assertion_pass_rate" not in fieldnames:
		raise RuntimeError(
			f"CSV missing required column 'assertion_pass_rate': {csv_path}"
		)

	rate_by_index: Dict[str, str] = {}
	rate_by_identity: Dict[Tuple[str, str, str], str] = {}
	for idx, candidate in enumerate(candidates_for_update, start=1):
		rate_text = str(candidate.assertion_pass_rate)
		rate_by_index[str(idx)] = rate_text
		identity = (
			candidate.method.strip().lower(),
			candidate.model.strip().lower(),
			_design_id_run_id(candidate).strip(),
		)
		rate_by_identity[identity] = rate_text

	for row in rows:
		identity = (
			str(row.get("method", "")).strip().lower(),
			str(row.get("model", "")).strip().lower(),
			str(row.get("design_id/run_id", "")).strip(),
		)
		if identity in rate_by_identity:
			row["assertion_pass_rate"] = rate_by_identity[identity]
			continue

		idx = str(row.get("index", "")).strip()
		if idx in rate_by_index:
			row["assertion_pass_rate"] = rate_by_index[idx]

	with csv_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

def _load_json(path: Path) -> Dict[str, object]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, dict):
		raise ValueError(f"Expected object JSON: {path}")
	return data


def _load_candidates(summary_files: Sequence[Path]) -> List[RunCandidate]:
	unique_candidates: Dict[Path, RunCandidate] = {}

	for summary_path in summary_files:
		try:
			summary = _load_json(summary_path)
		except Exception:
			continue

		method = str(summary.get("method", "")).strip()
		model = str(summary.get("model", "")).strip()
		design_id = str(summary.get("design_id", "")).strip()
		runs = summary.get("runs", [])
		if not isinstance(runs, list):
			continue

		for run in runs:
			if not isinstance(run, dict):
				continue
			result_file = str(run.get("result_file", "")).strip()
			if not result_file:
				continue

			run_id_value = run.get("run_id")
			run_id: Optional[int] = run_id_value if isinstance(run_id_value, int) else None
			if run_id is None:
				m = re.search(r"run(\d+)", result_file, re.IGNORECASE)
				if m:
					run_id = int(m.group(1))

			assertion_pass_rate = _safe_float(run.get("assertion_pass_rate"))
			if assertion_pass_rate is None:
				continue

			result_path = (summary_path.parent / result_file).resolve()
			if not result_path.exists() or not result_path.is_file():
				continue

			candidate = RunCandidate(
				result_path=result_path,
				method=method,
				model=model,
				design_id=design_id,
				run_id=run_id,
				assertion_pass_rate=max(0.0, min(1.0, assertion_pass_rate)),
			)
			unique_candidates[result_path] = candidate

	return sorted(unique_candidates.values(), key=lambda x: str(x.result_path))


def _bin_index(assertion_pass_rate: float) -> int:
	for idx, (_, lower, upper, _) in enumerate(BIN_SPECS):
		if lower <= assertion_pass_rate < upper:
			return idx
	return len(BIN_SPECS) - 1


def _allocate_targets(total_samples: int) -> List[int]:
	raw = [total_samples * ratio for _, _, _, ratio in BIN_SPECS]
	base = [int(math.floor(x)) for x in raw]
	remain = total_samples - sum(base)

	fractional_rank = sorted(
		range(len(raw)),
		key=lambda i: (raw[i] - base[i], -i),
		reverse=True,
	)
	for i in range(remain):
		base[fractional_rank[i]] += 1
	return base


def _stratified_sample(
	candidates: Sequence[RunCandidate],
	total_samples: int,
	seed: int,
) -> Tuple[List[RunCandidate], List[int], List[int]]:
	if total_samples <= 0:
		return [], [0] * len(BIN_SPECS), [0] * len(BIN_SPECS)

	bins: List[List[RunCandidate]] = [[] for _ in BIN_SPECS]
	for c in candidates:
		bins[_bin_index(c.assertion_pass_rate)].append(c)

	for bucket in bins:
		bucket.sort(key=lambda x: str(x.result_path))

	targets = _allocate_targets(total_samples)
	rng = random.Random(seed)

	picked: List[RunCandidate] = []
	selected_paths: set[Path] = set()
	actual_counts = [0] * len(BIN_SPECS)

	for idx, bucket in enumerate(bins):
		k = min(targets[idx], len(bucket))
		if k == 0:
			continue
		chosen = rng.sample(bucket, k)
		picked.extend(chosen)
		actual_counts[idx] = len(chosen)
		selected_paths.update(item.result_path for item in chosen)

	requested = min(total_samples, len(candidates))
	if len(picked) < requested:
		leftovers = [c for c in candidates if c.result_path not in selected_paths]
		extra_k = min(requested - len(picked), len(leftovers))
		if extra_k > 0:
			extra = rng.sample(leftovers, extra_k)
			picked.extend(extra)

	rng.shuffle(picked)
	return picked, targets, actual_counts


def _sanitize_payload(payload: Dict[str, object]) -> Tuple[Dict[str, object], str, str, str]:
	cleaned = dict(payload)

	removed_method = ""
	removed_model = ""
	design_id = ""

	meta = cleaned.get("meta")
	if isinstance(meta, dict):
		meta_clean = dict(meta)
		removed_method = str(meta_clean.pop("method", "")).strip()
		removed_model = str(meta_clean.pop("model", "")).strip()
		meta_clean.pop("provider", None)
		meta_clean.pop("wall_time_seconds", None)
		design_id = str(meta_clean.get("design_id", "")).strip()
		cleaned["meta"] = meta_clean

	cleaned.pop("compile_result", None)
	cleaned.pop("graph_snapshot", None)
	return cleaned, removed_method, removed_model, design_id


def _prepare_output_dir(sample_dir: Path) -> None:
	sample_dir.mkdir(parents=True, exist_ok=True)
	for old_file in sample_dir.glob("sample*.json"):
		if old_file.is_file():
			old_file.unlink()


def _write_outputs(
	sampled: Sequence[RunCandidate],
	sample_dir: Path,
	csv_path: Path,
) -> None:
	rows: List[List[object]] = []

	for idx, candidate in enumerate(sampled, start=1):
		payload = _load_json(candidate.result_path)
		sanitized, method_removed, model_removed, design_id = _sanitize_payload(payload)

		if not method_removed:
			method_removed = candidate.method
		if not model_removed:
			model_removed = candidate.model
		if not design_id:
			design_id = candidate.design_id

		design_id_run_id = design_id
		if candidate.run_id is not None:
			design_id_run_id = f"{design_id}/{candidate.run_id}"

		out_path = sample_dir / f"sample{idx}.json"
		with out_path.open("w", encoding="utf-8") as f:
			json.dump(sanitized, f, ensure_ascii=False, indent=2)

		rows.append(
			[
				idx,
				method_removed,
				model_removed,
				design_id_run_id,
				candidate.assertion_pass_rate,
				"",
				"",
				"",
			]
		)

	csv_path.parent.mkdir(parents=True, exist_ok=True)

	with csv_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(
			[
				"index",
				"method",
				"model",
				"design_id/run_id",
				"assertion_pass_rate",
				"expert1",
				"expert2",
				"final_decision",
			]
		)
		writer.writerows(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Stratified sampling for expert validation.")
	parser.add_argument("--total-samples", type=int, default=TOTAL_SAMPLES)
	parser.add_argument("--seed", type=int, default=SEED)
	parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
	parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
	return parser


def main() -> None:
	args = _build_arg_parser().parse_args()

	results_root = _resolve_path_arg(args.results_root)
	output_root = _resolve_path_arg(args.output_root)
	sample_dir = output_root / "validation_sample"
	csv_path = output_root / "validation_results.csv"

	summary_files = _discover_summary_files(results_root)
	candidates = _load_candidates(summary_files)
	if not candidates:
		raise RuntimeError(f"No run candidates found from {results_root}")

	if csv_path.exists():
		_update_existing_csv_assertion_pass_rate(
			csv_path=csv_path,
			candidates_for_update=candidates,
		)
		print(f"[sampling] candidates={len(candidates)}")
		print(f"[sampling] updated existing csv assertion_pass_rate only: {csv_path}")
		return

	sampled, targets, actual = _stratified_sample(
		candidates=candidates,
		total_samples=int(args.total_samples),
		seed=int(args.seed),
	)
	if not sampled:
		raise RuntimeError("Sampling returned zero items.")

	_prepare_output_dir(sample_dir)
	_write_outputs(sampled=sampled, sample_dir=sample_dir, csv_path=csv_path)

	print(f"[sampling] candidates={len(candidates)} sampled={len(sampled)}")
	print(f"[sampling] output_dir={sample_dir}")
	print(f"[sampling] csv={csv_path}")
	for idx, (label, _, _, _) in enumerate(BIN_SPECS):
		print(
			f"[sampling] bin {label}: target={targets[idx]} actual={actual[idx]}"
		)


if __name__ == "__main__":
	main()
