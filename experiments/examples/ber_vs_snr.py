"""Waterfall curve: image BER / PSNR vs channel SNR, per bits_per_symbol.

Run from the repo root:  python experiments/examples/ber_vs_snr.py
Writes results/ber_vs_snr.csv and results/ber_vs_snr.png.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments import ExperimentConfig, channel, metrics, save_csv, sweep
from experiments.plotting import plot_metric_vs_param

# 26 s: a full 60x60x3 image at the default rate needs ~22 s plus sync.
base = ExperimentConfig(payload_kind="image", f0=400.0, duration_s=26.0)

rows = sweep(
    base,
    grid={
        "bits_per_symbol": [1, 2, 4],
        "channel": [channel.AWGN(snr_db=s) for s in (40, 30, 25, 20, 15, 10)],
    },
    collect={
        "snr_db": lambda r: r.config.channel.snr_db,
        "ber": lambda r: metrics.bit_error_rate(
            r.ground_truth_image[0], r.last_image[0] if r.last_image else None),
        "psnr_db": lambda r: metrics.image_psnr_db(r.ground_truth_image, r.last_image),
        "ssim": lambda r: metrics.image_ssim(r.ground_truth_image, r.last_image),
    },
)

out = REPO_ROOT / "results"
save_csv(rows, str(out / "ber_vs_snr.csv"))
plot_metric_vs_param(rows, x="snr_db", y="ber", group_by="bits_per_symbol",
                     save=str(out / "ber_vs_snr.png"))
print(f"wrote {out / 'ber_vs_snr.csv'} and {out / 'ber_vs_snr.png'}")
