from math import gcd

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from Payload import Payload, SerializedPayload
from Payload.AudioPayload import AudioPayload
from Payload.pixel_codec import AudioDigitalCodec
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

        if self._serializer_mode == SerializerMode.DIGITAL:
            samples = self._quantize(samples)

        self._serialized_payload = SerializedPayload(samples)

    def _quantize(self, samples: list) -> list:
        codec = AudioDigitalCodec(
            self._settings.bits_per_symbol, self._settings.audio_samples_per_symbol,
        )
        step = codec.chunk_size
        data: list = []
        for offset in range(0, len(samples), step):
            data.extend(codec.encode_chunk(samples[offset:offset + step]))
        return data

    def _resample(self, samples: list, native_rate: int) -> list:
        target_rate = int(self._settings.MSG_FS)
        if self._serializer_mode == SerializerMode.DIGITAL:
            # samples_per_symbol raw samples share one symbol's time slot, so
            # the source must be resampled that much denser to fill them
            # without changing playback duration/speed.
            target_rate *= self._settings.audio_samples_per_symbol

        data = np.array(samples, dtype=np.float32)

        # resample_poly's built-in anti-alias filter is tuned for
        # general-purpose resampling, not for rejecting a wideband source
        # (e.g. a full-range sweep) whose content extends well past the
        # target Nyquist. Pre-filtering with a steep, dedicated lowpass at
        # the theoretical Nyquist prevents that content from folding back
        # (aliasing) into the passband during the downsample below.
        nyquist = target_rate / 2.0
        native_nyquist = native_rate / 2.0
        if nyquist < native_nyquist:
            sos = butter(8, nyquist / native_nyquist, btype="low", output="sos")
            data = sosfiltfilt(sos, data).astype(np.float32)

        divisor = gcd(native_rate, target_rate)
        up = target_rate // divisor
        down = native_rate // divisor
        resampled = resample_poly(data, up, down)
        return resampled.tolist()
