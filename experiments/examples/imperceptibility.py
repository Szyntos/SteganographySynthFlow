"""Rate vs audibility: embedding SNR and payload BER as phase_range grows.

Run from the repo root:  python experiments/examples/imperceptibility.py
"""

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments import (ExperimentConfig, metrics, render_carrier,
                         run_experiment, save_csv)

rows = []
for phase_range in (math.pi / 32, math.pi / 16, math.pi / 8, math.pi / 4):
    # 26 s: a full 60x60x3 image at 4000 bps raw needs ~22 s plus sync,
    # otherwise BER measures truncation, not the channel.
    cfg = ExperimentConfig(payload_kind="image", f0=400.0, duration_s=26.0,
                           settings_overrides={"phase_range": phase_range})
    r = run_experiment(cfg)
    carrier = render_carrier(cfg, num_samples=len(r.encoded))
    rows.append({
        "phase_range": round(phase_range, 4),
        "embedding_snr_db": round(metrics.embedding_snr_db(carrier, r.encoded), 2),
        "lsd_db": round(metrics.log_spectral_distance_db(carrier, r.encoded), 2),
        "ber": metrics.bit_error_rate(
            r.ground_truth_image[0], r.last_image[0] if r.last_image else None),
    })
    print(rows[-1])

save_csv(rows, str(REPO_ROOT / "results" / "imperceptibility.csv"))
