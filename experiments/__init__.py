"""Measurement harness for thesis experiments.

Wraps the real EncoderDSP/DecoderDSP facades in a headless, scriptable
pipeline:

    config -> encode -> channel (impairments) -> decode -> RunResult -> metrics

Typical use::

    from experiments import ExperimentConfig, run_experiment, channel, metrics

    cfg = ExperimentConfig(payload_kind="image", f0=400.0,
                           channel=channel.Chain(channel.AWGN(snr_db=30)))
    result = run_experiment(cfg)
    print(metrics.image_psnr_db(result.ground_truth_image, result.last_image))

See experiments/README.md for the catalogue of suggested analyses.
"""

from .config import ExperimentConfig
from .runner import RunResult, run_experiment, render_carrier
from .sweep import sweep, save_csv
from . import channel, metrics

__all__ = [
    "ExperimentConfig", "RunResult", "run_experiment", "render_carrier",
    "sweep", "save_csv", "channel", "metrics",
]
