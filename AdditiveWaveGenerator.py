from typing import List

class AdditiveWaveGenerator:
    def __init__(self):
        self._omegas: List[float] = []
        self._phases: List[float] = []
        self._chunk_size: int = 0

    def advance_phases(self) -> None:
        pass

    def set_chunk_size(self, chunk_size: int):
        self._chunk_size = chunk_size

    def set_omegas(self, omegas: List[float]) -> None:
        self._omegas = omegas

    def set_phases(self, phases: List[float]) -> None:
        self._phases = phases

    def get_omegas(self) -> List[float]:
        return self._omegas

    def get_phases(self) -> List[float]:
        self.advance_phases()
        return self._phases