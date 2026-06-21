from AudioChunk import AudioChunk
from Encoder import Encoder


class AudioDevice:
    def __init__(self):
        pass
    def audioCallback(self, encoder: Encoder, num_samples: int) -> AudioChunk:
        return encoder.process(num_samples)