from typing import List


class SerializedPayload:
    def __init__(self):
        self._offsets: List[float] = []

    def get_offsets(self) -> List[float]:
        return self._offsets