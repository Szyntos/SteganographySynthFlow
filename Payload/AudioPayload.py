from Data import Data

from AudioChunk import AudioChunk
from Payload import Payload


class AudioPayload(Payload):
    def __init__(self, size: int):
        super().__init__(size)

    def set_data(self, data: Data):
        pass

    def get_data(self) -> Data:
        pass

    def to_audio_chunk(self) -> AudioChunk:
        return AudioChunk([i**2 for i in range(self._size)])
