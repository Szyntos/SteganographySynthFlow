from collections import deque
from typing import List, Optional

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from Framing import FramingSyncController
from Payload import SymbolRow
from Payload.pixel_codec import AudioDigitalCodec
from SerializerMode import SerializerMode
from Settings import Settings
from .Sink import Sink
from .SinkBehaviour import SinkBehaviour

# MP3 (MPEG-1/2/2.5) only supports these sample rates.
MP3_SAMPLE_RATES: List[int] = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]


class AudioSink(Sink):
    """Records the decoder's raw sample stream into a rolling buffer capped
    at max_buffer_seconds, so long recordings can't grow without bound.
    Dumping to a file empties the buffer."""

    def __init__(self,
                 framing_sync_controller: FramingSyncController,
                 sink_behaviour: SinkBehaviour,
                 settings: Optional[Settings] = None,
                 sample_rate: Optional[int] = None,
                 max_buffer_seconds: Optional[float] = None,
                 serializer_mode: SerializerMode = SerializerMode.ANALOGUE):
        super().__init__(framing_sync_controller, sink_behaviour)
        if sample_rate is None and settings is None:
            raise ValueError("AudioSink: provide either settings or an explicit sample_rate")
        if sample_rate is not None:
            self._sample_rate = int(sample_rate)
        else:
            self._sample_rate = int(settings.MSG_FS)
            if serializer_mode == SerializerMode.DIGITAL:
                # Mirrors AudioSerializer._resample: samples_per_symbol raw
                # samples share one symbol's time slot, so playback needs a
                # denser sample rate to keep the same duration/speed.
                self._sample_rate *= settings.audio_samples_per_symbol
        buffer_seconds = (
            max_buffer_seconds if max_buffer_seconds is not None
            else (settings.sink_max_buffer_seconds if settings is not None else 120.0)
        )
        max_samples = int(self._sample_rate * buffer_seconds)
        self._buffer: deque = deque(maxlen=max_samples)

        self._audio_codec: Optional[AudioDigitalCodec] = None
        if serializer_mode == SerializerMode.DIGITAL:
            if settings is None:
                raise ValueError("AudioSink: digital mode requires settings")
            self._audio_codec = AudioDigitalCodec(
                settings.bits_per_symbol, settings.audio_samples_per_symbol,
            )

    def push(self, symbol_row: SymbolRow) -> None:
        offsets = symbol_row.get_offsets()
        if self._audio_codec is None:
            self._buffer.extend(offsets)
            return
        for level in offsets:
            self._buffer.extend(self._audio_codec.decode_chunk([level]))

    def push_many(self, symbol_rows: List[SymbolRow]) -> None:
        for symbol_row in symbol_rows:
            self.push(symbol_row)

    def get_buffer_duration_seconds(self) -> float:
        return len(self._buffer) / self._sample_rate

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def clear(self) -> None:
        self._buffer.clear()

    def dump_to_wav(self, file_path: str) -> None:
        self._dump(file_path, "WAV")

    def dump_to_mp3(self, file_path: str) -> None:
        samples = np.fromiter(self._buffer, dtype=np.float32, count=len(self._buffer))
        target_rate = min(MP3_SAMPLE_RATES, key=lambda r: abs(r - self._sample_rate))
        if target_rate != self._sample_rate:
            divisor = gcd(self._sample_rate, target_rate)
            up = target_rate // divisor
            down = self._sample_rate // divisor
            samples = resample_poly(samples, up, down).astype(np.float32)
        sf.write(file_path, samples, target_rate, format="MP3")
        self.clear()

    def _dump(self, file_path: str, fmt: str) -> None:
        samples = np.fromiter(self._buffer, dtype=np.float32, count=len(self._buffer))
        sf.write(file_path, samples, self._sample_rate, format=fmt)
        self.clear()
