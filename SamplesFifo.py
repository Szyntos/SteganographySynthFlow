from typing import List


class SamplesFifo:
    def __init__(self, startup_threshold: int = 0):
        self._samples: List[float] = []
        self._startup_threshold: int = startup_threshold
        self._started: bool = startup_threshold == 0

    def get_size(self) -> int:
        return len(self._samples)

    def has_started(self) -> bool:
        return self._started

    def can_read(self, n: int) -> bool:
        return self._started and len(self._samples) >= n

    def push(self, samples: List[float]) -> None:
        self._samples += samples

        if not self._started and len(self._samples) >= self._startup_threshold:
            self._started = True

    def pop(self, n: int) -> List[float]:
        if not self.can_read(n):
            raise RuntimeError(
                f"Cannot pop {n} samples. "
                f"Available: {len(self._samples)}, "
                f"started: {self._started}"
            )

        result = self._samples[:n]
        self._samples = self._samples[n:]
        return result

    def pop_or_empty(self, n: int) -> List[float]:
        if self.can_read(n):
            return self.pop(n)

        return []

    def pop_or_silence(self, n: int) -> List[float]:
        if self.can_read(n):
            return self.pop(n)

        return [0.0] * n