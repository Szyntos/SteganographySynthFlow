from AudioChunk import AudioChunk
from Payload import Payload


class AudioPayload(Payload):
    def __init__(self):
        super().__init__()

    def to_audio_chunk(self) -> AudioChunk:
        return AudioChunk()
