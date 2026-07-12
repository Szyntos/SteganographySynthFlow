from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitLayout:
    phases: int
    pilot_start: int
    data_start: int
    envelope: np.ndarray

    @staticmethod
    def two_split(internal_clock: int) -> "SplitLayout":
        half = internal_clock // 2
        envelope = np.zeros(internal_clock, dtype=np.float64)
        envelope[half:] = 1.0
        return SplitLayout(phases=2, pilot_start=0, data_start=half, envelope=envelope)

    @staticmethod
    def four_split(internal_clock: int) -> "SplitLayout":
        quarter = internal_clock // 4
        envelope = SplitLayout._build_envelope(quarter, internal_clock)
        return SplitLayout(phases=4, pilot_start=0, data_start=2 * quarter, envelope=envelope)

    @staticmethod
    def _build_envelope(q: int, chunk_size: int) -> np.ndarray:
        env = np.zeros(chunk_size, dtype=np.float64)
        if q == 0:
            return env

        n = np.arange(q, dtype=np.float64)
        if q > 1:
            env[q:2 * q] = n / q
        else:
            env[q:2 * q] = 1.0

        env[2 * q:3 * q] = 1.0

        if q > 1:
            env[3 * q:4 * q] = 1.0 - n / q
        else:
            env[3 * q:4 * q] = 0.0

        return env
