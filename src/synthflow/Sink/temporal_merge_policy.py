from typing import Optional, Sequence, Tuple


class TemporalMergePolicy:
    """Clean-mode temporal blending of finalized frames into a persistent image.

    Called exactly once per finalized/dropped frame. Live mode never uses this
    class; it publishes the canvas directly on every write.
    """

    def __init__(self,
                 blend_n: float = 0.85,
                 replace_similarity_threshold: float = 0.51,
                 replace_min_coverage: float = 0.25,
                 similarity_scale: float = 255.0):
        self._blend_n = blend_n
        self._replace_similarity_threshold = replace_similarity_threshold
        self._replace_min_coverage = replace_min_coverage
        self._similarity_scale = similarity_scale
        self.persist: Optional[bytearray] = None

    def reset(self) -> None:
        self.persist = None

    def compute_similarity_and_coverage(self,
                                        canvas: Sequence[int],
                                        arrived: Sequence[int]) -> Tuple[float, float]:
        """Mean-absolute-difference similarity over arrived bytes only.

        Returns (-1.0, coverage) if there is no persist yet or nothing arrived.
        """
        total = len(canvas)
        arrived_count = sum(1 for flag in arrived if flag)
        coverage = 0.0 if total == 0 else arrived_count / total

        if self.persist is None or arrived_count == 0:
            return -1.0, coverage

        sum_abs = 0
        for i in range(total):
            if not arrived[i]:
                continue
            sum_abs += abs(canvas[i] - self.persist[i])

        mad = sum_abs / arrived_count
        similarity = 1.0 - mad / self._similarity_scale
        similarity = min(max(similarity, 0.0), 1.0)
        return similarity, coverage

    def merge(self, canvas: Sequence[int], arrived: Sequence[int]) -> None:
        if not any(arrived):
            return

        if self.persist is None:
            # First frame: adopt the canvas directly — never blend against a
            # phantom zero image.
            self.persist = bytearray(canvas)
            return

        similarity, coverage = self.compute_similarity_and_coverage(canvas, arrived)

        if (0.0 <= similarity < self._replace_similarity_threshold
                and coverage >= self._replace_min_coverage):
            self.persist = bytearray(canvas)
            return

        n = self._blend_n
        for i in range(len(canvas)):
            if not arrived[i]:
                continue
            blended = n * self.persist[i] + (1.0 - n) * canvas[i]
            self.persist[i] = min(255, max(0, int(blended + 0.5)))
