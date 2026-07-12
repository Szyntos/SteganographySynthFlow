"""Offline encode→decode harness.

Runs the full pipeline headlessly against a payload at a fixed f0, with no
audio device and no GUI, so parameter sweeps and plots can be driven from a
plain script. Each run gets its own Settings, so configs never alias.
"""

import sys
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from AdditiveWaveGenerator import AdditiveWaveGenerator
from Decoder import Decoder
from Encoder import Encoder
from Framing import FramingSyncController
from Payload import AudioPayload, Payload, SymbolRow
from Serializer import AudioSerializer
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import AudioSink, SinkBehaviour
from StrategyKinds import (
    DECODING_STRATEGY_CLASSES,
    ENCODING_STRATEGY_CLASSES,
    apply_strategy_kind,
)


@dataclass
class RoundTrip:
    """One offline encode→decode run."""

    encoded: List[float]
    decoded: List[float]
    expected: List[float]
    startup_lag: int
    settings: Settings = field(repr=False)

    @property
    def aligned_decoded(self) -> List[float]:
        """The decoded samples that line up with `expected`."""
        return self.decoded[self.startup_lag: self.startup_lag + len(self.expected)]

    @property
    def diff(self) -> np.ndarray:
        """Decoded − expected, over the aligned region."""
        return np.asarray(self.aligned_decoded) - np.asarray(self.expected)

    @property
    def diff_dc_removed(self) -> np.ndarray:
        d = self.diff
        return d - d.mean()

    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.diff ** 2)))


def compute_startup_lag(settings: Settings, num_samples: int) -> int:
    """Samples of silence the decoder emits before its FIFOs are primed."""
    chunk_size = settings.chunk_size
    startup_threshold = chunk_size + settings.max_driver_block_size - 1
    input_fifo = 0
    output_fifo = 0
    silence = 0
    while True:
        input_fifo += num_samples
        while input_fifo >= chunk_size:
            input_fifo -= chunk_size
            output_fifo += chunk_size
        if output_fifo >= startup_threshold:
            break
        silence += num_samples
    return silence


def expected_decoded_signal(payload: AudioPayload, settings: Settings, num_samples: int) -> List[float]:
    """The signal a perfect decoder would emit for this payload: the payload
    resampled to the symbol rate, then stretched back out to chunk length."""
    raw = payload.get_data()
    if payload.get_sample_rate() > 0:
        native_rate = payload.get_sample_rate()
        target_rate = int(settings.MSG_FS)
        divisor = gcd(native_rate, target_rate)
        raw = resample_poly(
            np.array(raw, dtype=np.float32), target_rate // divisor, native_rate // divisor
        ).tolist()

    data_harmonics = settings.data_harmonics
    chunk_size = settings.chunk_size
    n = len(raw)
    result: List[float] = []
    i = 0
    while len(result) < num_samples:
        chunk = [raw[(i + j) % n] for j in range(data_harmonics)]
        result += SymbolRow(chunk).resample_to_size(chunk_size)
        i = (i + data_harmonics) % n
    return result[:num_samples]


def run_round_trip(
    payload: Optional[Payload] = None,
    settings: Optional[Settings] = None,
    f0: float = 500.0,
    num_chunks: int = 160,
    strategy_kind: str = "two",
    codec_mode: SerializerMode = SerializerMode.DIGITAL,
    block_size: Optional[int] = None,
) -> RoundTrip:
    """Encode `payload` at `f0` and decode it straight back, in-process.

    `settings` is mutated to match `strategy_kind` (chunk size), so pass a
    fresh Settings per configuration rather than sharing one across a sweep.
    """
    settings = settings if settings is not None else Settings()
    apply_strategy_kind(settings, strategy_kind)
    settings.validate()

    if payload is None:
        payload = AudioPayload()
        payload.load_from_file(settings.modulator_wav_path)

    serializer = AudioSerializer(settings, codec_mode)
    encoding_strategy = ENCODING_STRATEGY_CLASSES[strategy_kind](
        settings, AdditiveWaveGenerator.harmonic(settings), serializer
    )
    encoding_strategy.load_payload(payload)
    encoder = Encoder(encoding_strategy)

    decoding_strategy = DECODING_STRATEGY_CLASSES[strategy_kind](settings)
    sink = AudioSink(FramingSyncController(), SinkBehaviour.LIVE, settings)
    decoder = Decoder(settings, decoding_strategy, sink)

    encoder.set_f0(f0)
    decoding_strategy.set_f0(f0)

    block_size = block_size or settings.audio_driver_polling_rate
    encoded: List[float] = []
    decoded: List[float] = []
    for _ in range(num_chunks):
        enc_chunk = encoder.process(block_size)
        encoded += enc_chunk.get_samples()
        decoded += decoder.process(enc_chunk, block_size).get_samples()

    startup_lag = compute_startup_lag(settings, block_size)
    expected = (
        expected_decoded_signal(payload, settings, len(decoded) - startup_lag)
        if isinstance(payload, AudioPayload)
        else []
    )
    return RoundTrip(encoded, decoded, expected, startup_lag, settings)
