from typing import List

from AudioChunk import AudioChunk


class SymbolRow:
    def __init__(self, offsets: List[float]):
        self._offsets: List[float] = offsets

    def get_size(self) -> int:
        return len(self._offsets)

    def get_offsets(self) -> List[float]:
        return self._offsets

    def resample_to_size(self, num_samples: int) -> List[float]:
        if num_samples <= 0:
            return []

        old_size = len(self._offsets)

        if old_size == 0:
            return []

        if old_size == 1:
            return [self._offsets[0]] * num_samples

        if old_size == num_samples:
            return self._offsets.copy()

        result: List[float] = []

        scale = (old_size - 1) / (num_samples - 1)

        for i in range(num_samples):
            pos = i * scale
            left = int(pos)
            right = min(left + 1, old_size - 1)
            t = pos - left

            value = (
                self._offsets[left] * (1.0 - t)
                + self._offsets[right] * t
            )
            result.append(value)

        return result