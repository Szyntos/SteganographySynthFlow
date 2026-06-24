from Data import Data
from abc import ABC, abstractmethod


class Payload(ABC):
    def __init__(self, size: int):
        self._size: int = size

    @abstractmethod
    def set_data(self, data: Data):
        pass

    @abstractmethod
    def get_data(self) -> Data:
        pass

    @abstractmethod
    def get_size(self):
        pass