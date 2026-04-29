from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def _normalize_vote(value: str) -> str:
    return value.strip().lower()


def _resolve_target_column(fieldnames: List[str]) -> str:
    if "final_decision" in fieldnames:
        return "final_decision"
    if "final_decsion" in fieldnames:
        return "final_decsion"
    return "final_decision"


def update_final_decision(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV has no header: {csv_path}")

        fieldnames = list(reader.fieldnames)
        target_col = _resolve_target_column(fieldnames)
        if target_col not in fieldnames:
            fieldnames.append(target_col)

        rows: List[Dict[str, str]] = [dict(row) for row in reader]

    updated_count = 0
    for row in rows:
        expert1 = _normalize_vote(row.get("expert1", ""))
        expert2 = _normalize_vote(row.get("expert2", ""))

        if expert1 in {"pass", "fail"} and expert2 in {"pass", "fail"} and expert1 == expert2:
            row[target_col] = expert1
            updated_count += 1

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return updated_count


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Update final_decision/final_decsion when expert1 and expert2 both "
            "have the same pass/fail vote."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="validation_results.csv",
        help="Path to validation CSV (default: validation_results.csv)",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()

    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    updated = update_final_decision(csv_path)
    print(f"[fill_final_decision] updated_rows={updated} csv={csv_path}")


if __name__ == "__main__":
    main()
