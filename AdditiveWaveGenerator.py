import math
from typing import List

from Payload import SymbolRow


class AdditiveWaveGenerator:
    def __init__(self):
        self._omegas: List[float] = []
        self._phases: List[float] = []
        self._amps: List[float] = []
        self._phase_offset_range = math.pi / 8.0

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
        if len(self._omegas) != len(self._phases):
            raise ValueError("omegas and phases must have the same length")

        if len(self._omegas) != len(self._amps):
            raise ValueError("omegas and amps must have the same length")

    def generate_next(self, f0: float) -> float:
        self._validate_state()

        sample: float = 0.0
        amp_sum: float = sum(abs(amp) for amp in self._amps)

        for i in range(len(self._omegas)):
            sample += self._amps[i] * math.sin(self._phases[i])

            self._phases[i] += self._omegas[i] * f0
            self._phases[i] %= 2.0 * math.pi

        if amp_sum > 0.0:
            sample /= amp_sum

        return sample

    def generate_next_with_offsets(self, f0: float, symbol_row: SymbolRow) -> float:
        self._validate_state()

        if symbol_row.get_size() > len(self._omegas):
            raise ValueError("SymbolRow contains more offsets than there are harmonics")

        sample: float = 0.0
        offsets: List[float] = symbol_row.get_offsets()
        amp_sum: float = sum(abs(amp) for amp in self._amps)

        first_offset_harmonic = len(self._omegas) - len(offsets)

        for i in range(len(self._omegas)):
            phase = self._phases[i]

            if i >= first_offset_harmonic:
                offset_index = i - first_offset_harmonic
                phase += offsets[offset_index] * self._phase_offset_range

            sample += self._amps[i] * math.sin(phase)

            self._phases[i] += self._omegas[i] * f0
            self._phases[i] %= 2.0 * math.pi

        if amp_sum > 0.0:
            sample /= amp_sum

        return sample