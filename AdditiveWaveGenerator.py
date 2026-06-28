from typing import List


class AdditiveWaveGenerator:
    def __init__(self):
        self._omegas: List[float] = []
        self._phases: List[float] = []

    def set_omegas(self, omegas: List[float]) -> None:
        self._omegas = omegas

    def set_phases(self, phases: List[float]) -> None:
        self._phases = phases

    def get_omegas(self) -> List[float]:
        return self._omegas

    def get_phases(self) -> List[float]:
        return self._phases
