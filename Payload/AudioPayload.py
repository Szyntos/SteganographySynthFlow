from typing import List

import numpy as np
import soundfile as sf

from Payload import Payload


class AudioPayload(Payload):
    def __init__(self):
        super().__init__()
        self._sample_rate: int = 0

    def load_from_file(self, file_path: str):
        samples, sample_rate = sf.read(file_path, dtype='float32', always_2d=True)
        mono = samples.mean(axis=1)
        self._data = mono.tolist()
        self._sample_rate = sample_rate

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def get_data(self) -> List[float]:
        # return [0.0] * len(self._data)
        return self._data
