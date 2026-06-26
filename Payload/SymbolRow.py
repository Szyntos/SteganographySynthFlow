from typing import List


class SymbolRow:
    def __init__(self, offsets: List[float]):
        self._offsets: List[float] = offsets

    def get_size(self) -> int:
        return len(self._offsets)

    def get_offsets(self) -> List[float]:
        return self._offsets