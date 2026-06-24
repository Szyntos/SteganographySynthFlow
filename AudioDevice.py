from AudioChunk import AudioChunk
from Encoder import Encoder


class AudioDevice:
    def __init__(self, encoder: Encoder):
        self._encoder = encoder

    def audioCallback(self, num_samples: int) -> AudioChunk:
        return self._encoder.process(num_samples)
