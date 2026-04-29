from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence


def save_csv(rows: Sequence[Dict[str, object]], output_path: str) -> None:
    """
    Save rows to CSV. Missing fields are auto-filled with empty strings.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def export_reports(rows: Sequence[Dict[str, object]], output_dir: str) -> None:
    """
    Export simplified CSV summaries only.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep design-level rows so that design_id is always present in main_results.csv.
    save_csv(rows, str(out_dir / "main_results.csv"))
