"""Parameter sweeps: run a grid of ExperimentConfigs and tabulate metrics.

    rows = sweep(
        base=ExperimentConfig(payload_kind="image", duration_s=8.0),
        grid={"bits_per_symbol": [1, 2, 3, 4],
              "channel": [channel.AWGN(snr_db=s) for s in (40, 30, 20, 10)]},
        collect={"psnr_db": lambda r: metrics.image_psnr_db(
                     r.ground_truth_image, r.last_image)},
    )
    save_csv(rows, "results/image_bps_vs_snr.csv")

Grid keys are ExperimentConfig field names; the cartesian product is run.
Each row also records every swept parameter (repr for non-scalars) plus
built-in bookkeeping columns.
"""

import csv
import itertools
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import ExperimentConfig
from .runner import RunResult, run_experiment

MetricFn = Callable[[RunResult], Any]


def _cell(value: Any) -> Any:
    return value if isinstance(value, (int, float, str, bool, type(None))) else repr(value)


def sweep(base: ExperimentConfig,
          grid: Dict[str, List[Any]],
          collect: Dict[str, MetricFn],
          repeats: int = 1,
          on_result: Optional[Callable[[Dict[str, Any], RunResult], None]] = None,
          verbose: bool = True) -> List[Dict[str, Any]]:
    """Run every combination in `grid` (times `repeats`), returning one row
    of {param..., metric...} per run. A crashing run yields a row with an
    'error' column instead of aborting the whole sweep."""
    names = list(grid.keys())
    rows: List[Dict[str, Any]] = []
    combos = list(itertools.product(*(grid[n] for n in names)))

    for combo in combos:
        params = dict(zip(names, combo))
        for rep in range(repeats):
            cfg = base.with_(**params)
            row: Dict[str, Any] = {n: _cell(v) for n, v in params.items()}
            row["repeat"] = rep
            if base.label:
                row["label"] = base.label
            try:
                result = run_experiment(cfg)
                row["first_frame_latency_s"] = result.first_frame_latency_s()
                row["realtime_factor"] = round(result.realtime_factor(), 4)
                row["gated_blocks"] = result.gated_blocks
                for metric_name, fn in collect.items():
                    row[metric_name] = _cell(fn(result))
                if on_result is not None:
                    on_result(row, result)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                if verbose:
                    traceback.print_exc()
            rows.append(row)
            if verbose:
                print(row)
    return rows


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
