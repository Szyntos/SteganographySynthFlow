from typing import List

from abc import ABC, abstractmethod


class Payload(ABC):
    def __init__(self):
        self._data: List[float] = []

    @abstractmethod
    def load_from_file(self, file_path: str):
        pass

    @abstractmethod
    def get_data(self) -> List[float]:
        pass