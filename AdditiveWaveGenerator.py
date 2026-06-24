from typing import List

from AudioChunk import AudioChunk


class AdditiveWaveGenerator:
    def __init__(self):
        self._omegas: List[float] = []
        self._phases: List[float] = []
        self._chunk_size: int = 0

    def set_chunk_size(self, chunk_size: int):
        self._chunk_size = chunk_size

    def set_omegas(self, omegas: List[float]) -> None:
        self._omegas = omegas

    def set_phases(self, phases: List[float]) -> None:
        self._phases = phases

    def get_omegas(self) -> List[float]:
        return self._omegas

    def get_phases(self) -> List[float]:
        return self._phases

    def generate(self) -> AudioChunk:
        pass  # synthesis: sum sinusoids at _omegas/_phases for _chunk_size samples

    def analyze(self, chunk: AudioChunk) -> List[float]:
        pass  # analysis: extract per-partial phases from chunk at _omegas
