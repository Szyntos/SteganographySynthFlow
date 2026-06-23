from AudioChunk import AudioChunk
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
    def to_audio_chunk(self) -> AudioChunk:
        return AudioChunk.silence(self._size)