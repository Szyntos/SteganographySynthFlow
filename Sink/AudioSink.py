from collections import deque
from typing import List, Optional

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from Framing import FramingSyncController
from Payload import SymbolRow
from Settings import Settings
from .Sink import Sink
from .SinkBehaviour import SinkBehaviour

MAX_BUFFER_SECONDS: float = 120.0

# MP3 (MPEG-1/2/2.5) only supports these sample rates.
MP3_SAMPLE_RATES: List[int] = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]


class AudioSink(Sink):
    """Records the decoder's raw sample stream into a rolling buffer capped
    at MAX_BUFFER_SECONDS, so long recordings can't grow without bound.
    Dumping to a file empties the buffer."""

    def __init__(self,
                 framing_sync_controller: FramingSyncController,
                 sink_behaviour: SinkBehaviour,
                 settings: Optional[Settings] = None,
                 sample_rate: Optional[int] = None):
        super().__init__(framing_sync_controller, sink_behaviour)
        self._sample_rate = int(sample_rate if sample_rate is not None else settings.MSG_FS)
        max_samples = int(self._sample_rate * MAX_BUFFER_SECONDS)
        self._buffer: deque = deque(maxlen=max_samples)

    def push(self, symbol_row: SymbolRow) -> None:
        self._buffer.extend(symbol_row.get_offsets())

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
