"""Experiment configuration: one immutable description of a full run.

Each run builds its own fresh Settings objects (one for the encoder, one for
the decoder) so sweeps never alias mutable state between configurations —
the same rule exp/harness.py already follows.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from SerializerMode import SerializerMode
from Settings import Settings

_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ExperimentConfig:
    # ── what is transmitted ──────────────────────────────────────────────
    payload_kind: str = "image"            # "audio" | "image" | "binary" | "text"
    payload_path: Optional[str] = None     # None -> facade default asset
    codec_mode: SerializerMode = SerializerMode.DIGITAL
    bits_per_symbol: int = 2

    # ── how it is transmitted ────────────────────────────────────────────
    strategy_kind: str = "two"             # "two" | "four"
    f0: float = 400.0                      # encoder carrier fundamental (Hz)

    # ── decoder-side configuration ───────────────────────────────────────
    decoder_f0: Optional[float] = None     # manual decoder f0; None -> same as f0
    f0_mode: str = "manual"                # "manual" | "autocorr" | "fft"
    pitch_quantize: bool = False
    # "clean" publishes once per finalized sync-delimited frame (what you
    # want for fidelity metrics); "live" publishes after every row write,
    # so the last publication may be a partial canvas.
    sink_behaviour: str = "clean"

    # ── run shape ────────────────────────────────────────────────────────
    duration_s: float = 10.0               # how much audio to synthesize
    block_size: Optional[int] = None       # None -> settings.audio_driver_polling_rate

    # ── channel between encoder and decoder ─────────────────────────────
    # Callable (np.ndarray, fs) -> np.ndarray; see experiments/channel.py.
    channel: Optional[Callable] = None

    # ── raw Settings overrides, applied to BOTH fresh Settings objects ──
    # e.g. {"phase_range": math.pi/16, "data_harmonics": 30, "image_target_w": 40}
    settings_overrides: Dict[str, Any] = field(default_factory=dict)

    # Free-form label carried into sweep result rows.
    label: str = ""

    def with_(self, **kwargs) -> "ExperimentConfig":
        """Functional update, for building sweep grids by hand."""
        return replace(self, **kwargs)

    def make_settings(self) -> Settings:
        settings = Settings()
        for name, value in self.settings_overrides.items():
            setattr(settings, name, value)
        # Default asset paths in Settings are relative to the repo root;
        # anchor them so experiments work from any working directory.
        for attr in ("modulator_wav_path", "image_path"):
            path = getattr(settings, attr)
            if not Path(path).is_absolute():
                setattr(settings, attr, str(_REPO_ROOT / path))
        return settings
