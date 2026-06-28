import math
from typing import List

from Payload import SymbolRow
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

    def _advance_phase(self, i: int, f0: float) -> None:
        self._phases[i] += 2.0 * math.pi * self._omegas[i] * f0 / self._sample_rate
        self._phases[i] %= 2.0 * math.pi

    def generate_next(self, f0: float) -> float:
        self._validate_state()

        sample: float = 0.0
        amp_sum: float = sum(abs(amp) for amp in self._amps)

        for i in range(len(self._omegas)):
            sample += self._amps[i] * math.sin(self._phases[i])
            self._advance_phase(i, f0)

        if amp_sum > 0.0:
            sample /= amp_sum

        return sample

    def generate_next_with_offsets(self, f0: float, symbol_row: SymbolRow) -> float:
        self._validate_state()

        offsets: List[float] = symbol_row.get_offsets()

        if len(offsets) > len(self._omegas):
            raise ValueError("SymbolRow contains more offsets than there are harmonics")

        sample: float = 0.0
        amp_sum: float = sum(abs(amp) for amp in self._amps)

        first_offset_harmonic: int = len(self._omegas) - len(offsets)

        for i in range(len(self._omegas)):
            phase: float = self._phases[i]

            if i >= first_offset_harmonic:
                offset_index: int = i - first_offset_harmonic
                phase += offsets[offset_index] * self._phase_offset_range

            sample += self._amps[i] * math.sin(phase)
            self._advance_phase(i, f0)

        if amp_sum > 0.0:
            sample /= amp_sum

        return sample
