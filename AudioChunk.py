from typing import List


class AudioChunk:
    def __init__(self, samples: List[float]):
        self._samples = samples

    @staticmethod
    def silence(size: int) -> "AudioChunk":
        return AudioChunk([0.01] * size)

    def get_samples(self) -> List[float]:
        return self._samples

    def size(self) -> int:
        return len(self._samples)

    def __str__(self) -> str:
        return f"AudioChunk{[i for i in self._samples]}"