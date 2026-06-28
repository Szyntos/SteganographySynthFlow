from math import gcd

import numpy as np
from scipy.signal import resample_poly

from Payload import Payload, SerializedPayload
from Payload.AudioPayload import AudioPayload
from SerializerMode import SerializerMode
from Settings import Settings
from .Serializer import Serializer


class AudioSerializer(Serializer):
    def __init__(self, settings: Settings, serializer_mode: SerializerMode):
        super().__init__(settings, serializer_mode)

    def load_payload(self, payload: Payload) -> None:
        self._payload = payload

        samples = payload.get_data()

        if isinstance(payload, AudioPayload) and payload.get_sample_rate() > 0:
            samples = self._resample(samples, payload.get_sample_rate())

        self._serialized_payload = SerializedPayload(samples)

    def _resample(self, samples: list, native_rate: int) -> list:
        target_rate = int(self._settings.MSG_FS)
        divisor = gcd(native_rate, target_rate)
        up = target_rate // divisor
        down = native_rate // divisor
        resampled = resample_poly(np.array(samples, dtype=np.float32), up, down)
        return resampled.tolist()
