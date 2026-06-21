from typing import List


class SerializedPayload:
    def __init__(self):
        self._samples: List[float] = []

    def get_samples(self) -> List[float]:
        return self._samples