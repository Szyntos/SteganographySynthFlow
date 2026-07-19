"""Additive-carrier parameter set: the editable wave shape and its exchange
format between the encoder and decoder sides.

The encoder needs all three arrays (amps, phases, omegas). The decoder's
demodulation is differential per harmonic, so amps and base phases cancel
out — but the frequency scalars (omegas) are baked into its DFT projection
matrices, so both sides must load the same file for the link to decode.
"""

import json
import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

from Settings import Settings

_FORMAT = "ssf-wave"
_VERSION = 1


@dataclass
class WaveParams:
    amps: List[float] = field(default_factory=list)
    phases: List[float] = field(default_factory=list)
    omegas: List[float] = field(default_factory=list)

    @classmethod
    def harmonic_default(cls, settings: Settings) -> "WaveParams":
        """Mirror of AdditiveWaveGenerator.harmonic: integer partials with
        1/n amplitudes and zero initial phases."""
        n = settings.total_harmonics
        return cls(
            amps=[settings.base_amplitude / (i + 1) for i in range(n)],
            phases=[0.0] * n,
            omegas=[float(i + 1) for i in range(n)],
        )

    def validate(self) -> None:
        if not (len(self.amps) == len(self.phases) == len(self.omegas)):
            raise ValueError("amps, phases and omegas must have the same length")
        if len(self.amps) == 0:
            raise ValueError("wave params must contain at least one harmonic")
        if any(w <= 0.0 for w in self.omegas):
            raise ValueError("omegas must be positive")
        if any(a < 0.0 for a in self.amps):
            raise ValueError("amps must be non-negative")

    def one_cycle(self, num_points: int) -> np.ndarray:
        """One cycle of the fundamental, evaluated at num_points, for display."""
        t = np.linspace(0.0, 1.0, num_points, endpoint=False)
        amps = np.asarray(self.amps, dtype=np.float64)
        phases = np.asarray(self.phases, dtype=np.float64)
        omegas = np.asarray(self.omegas, dtype=np.float64)
        return (amps[:, None]
                * np.sin(2.0 * math.pi * np.outer(omegas, t) + phases[:, None])).sum(axis=0)

    def to_json_file(self, file_path: str) -> None:
        payload = {
            "format": _FORMAT,
            "version": _VERSION,
            "total_harmonics": len(self.amps),
            "amps": list(self.amps),
            "phases": list(self.phases),
            "omegas": list(self.omegas),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_json_file(cls, file_path: str) -> "WaveParams":
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("format") != _FORMAT:
            raise ValueError("Not an SSF wave file (missing 'format': 'ssf-wave')")
        params = cls(
            amps=[float(a) for a in payload["amps"]],
            phases=[float(p) for p in payload["phases"]],
            omegas=[float(w) for w in payload["omegas"]],
        )
        params.validate()
        return params
