import math
from typing import List, Optional

import numpy as np

from Settings import Settings


class AdditiveWaveGenerator:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._sample_rate: float = 0.0
        self._phase_offset_range: float = 0.0
        self._omegas: List[float] = []
        self._phases: List[float] = []
        self._amps: List[float] = []
        self.reconfigure()

    def reconfigure(self) -> None:
        self._sample_rate = self._settings.fs_out
        self._phase_offset_range = self._settings.phase_range

    def set_omegas(self, omegas: List[float]) -> None:
        self._omegas = omegas

    def set_phases(self, phases: List[float]) -> None:
        self._phases = phases

    def set_amps(self, amps: List[float]) -> None:
        self._amps = amps

    def get_omegas(self) -> List[float]:
        return self._omegas

    def get_phases(self) -> List[float]:
        return self._phases

    def get_amps(self) -> List[float]:
        return self._amps

    def _validate_state(self) -> None:
        if self._sample_rate <= 0.0:
            raise ValueError("sample_rate must be positive")

        if len(self._omegas) != len(self._phases):
            raise ValueError("omegas and phases must have the same length")

        if len(self._omegas) != len(self._amps):
            raise ValueError("omegas and amps must have the same length")

    def generate_next(self, f0: float) -> float:
        return float(self.generate_block(f0, 1)[0])

    def generate_next_with_offsets(
        self,
        f0: float,
        phase_offsets: Optional[List[float]] = None,
        amp_offsets: Optional[List[float]] = None,
    ) -> float:
        return float(self.generate_block_with_offsets(f0, 1, phase_offsets, amp_offsets)[0])

    def generate_block(self, f0: float, n: int) -> np.ndarray:
        return self.generate_block_with_offsets(f0, n)

    def generate_block_with_offsets(
        self,
        f0: float,
        n: int,
        phase_offsets: Optional[List[float]] = None,
        amp_offsets: Optional[List[float]] = None,
        phase_envelope: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._validate_state()

        if phase_offsets is not None and len(phase_offsets) > len(self._omegas):
            raise ValueError("phase_offsets length exceeds number of harmonics")
        if amp_offsets is not None and len(amp_offsets) > len(self._omegas):
            raise ValueError("amp_offsets length exceeds number of harmonics")
        if phase_envelope is not None and len(phase_envelope) != n:
            raise ValueError("phase_envelope length must equal n")

        if n <= 0:
            return np.zeros(0, dtype=np.float64)

        omegas = np.asarray(self._omegas, dtype=np.float64)
        phases = np.asarray(self._phases, dtype=np.float64)
        amps = np.asarray(self._amps, dtype=np.float64)
        num_harmonics = len(omegas)

        # Per-harmonic phase advance per sample tick.
        deltas = 2.0 * math.pi * omegas * f0 / self._sample_rate

        # phase_matrix[i, k] = phase of harmonic i at sample k (before offsets).
        k = np.arange(n, dtype=np.float64)
        phase_matrix = phases[:, None] + np.outer(deltas, k)

        if phase_offsets is not None:
            padded_phase_offsets = np.zeros(num_harmonics, dtype=np.float64)
            padded_phase_offsets[:len(phase_offsets)] = phase_offsets
            offset_scale = (padded_phase_offsets * self._phase_offset_range)[:, None]
            if phase_envelope is not None:
                phase_matrix += offset_scale * np.asarray(phase_envelope, dtype=np.float64)[None, :]
            else:
                phase_matrix += offset_scale

        if amp_offsets is not None:
            padded_amp_offsets = np.zeros(num_harmonics, dtype=np.float64)
            padded_amp_offsets[:len(amp_offsets)] = amp_offsets
            effective_amps = amps + padded_amp_offsets
        else:
            effective_amps = amps

        # Mute harmonics that would alias past Nyquist at this f0.
        nyquist = self._sample_rate / 2.0
        valid_mask = (omegas * f0) <= nyquist
        effective_amps = np.where(valid_mask, effective_amps, 0.0)

        samples = (effective_amps[:, None] * np.sin(phase_matrix)).sum(axis=0)

        # Persistent carrier phase advances by n ticks, independent of the
        # per-call offsets (offsets only affect the sample evaluation above).
        new_phases = (phases + n * deltas) % (2.0 * math.pi)
        self._phases = new_phases.tolist()

        return samples
