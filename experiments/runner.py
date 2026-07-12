"""Headless encode -> channel -> decode runner over the real DSP facades.

Uses EncoderDSP and DecoderDSP (the exact objects the GUIs drive), so every
measurement includes the full production pipeline: serializer/framing on the
way in; FIFOs, energy gate, f0 estimation, drop tolerance, sync controller
and sinks on the way out.
"""

import time
from dataclasses import dataclass, field
from math import gcd
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.signal import resample_poly

from AdditiveWaveGenerator import AdditiveWaveGenerator
from DecoderDSP import DecoderDSP
from Encoder import Encoder
from EncoderDSP import EncoderDSP
from Payload import AudioPayload, SymbolRow
from Serializer import AudioSerializer
from SerializerMode import SerializerMode
from StrategyKinds import ENCODING_STRATEGY_CLASSES

from .config import ExperimentConfig

# (pixels, width, height, channels) — as published by ImageSink
ImageFrame = Tuple[bytes, int, int, int]


@dataclass
class RunResult:
    """Everything one run produced, ready for the metrics module."""

    config: ExperimentConfig
    fs: int
    chunk_size: int

    # signals
    encoded: np.ndarray                  # clean encoder output
    received: np.ndarray                 # after channel impairments
    decoded_audio: np.ndarray            # decoder's audio-domain output

    # framed sink publications, each tagged with the input-sample index at
    # which it was published (for sync-acquisition latency).
    images: List[Tuple[int, ImageFrame]] = field(default_factory=list)
    raw_images: List[Tuple[int, ImageFrame]] = field(default_factory=list)
    texts: List[Tuple[int, str]] = field(default_factory=list)
    datas: List[Tuple[int, bytes]] = field(default_factory=list)

    # ground truth (whichever matches config.payload_kind is set)
    ground_truth_image: Optional[ImageFrame] = None
    ground_truth_text: Optional[str] = None
    ground_truth_bytes: Optional[bytes] = None
    expected_audio: Optional[np.ndarray] = None   # aligned ideal decoder output

    # per-block decoder telemetry
    f0_track: List[float] = field(default_factory=list)
    confidence_track: List[float] = field(default_factory=list)
    gated_blocks: int = 0

    encode_wall_s: float = 0.0
    decode_wall_s: float = 0.0

    # ── convenience views ────────────────────────────────────────────────
    @property
    def last_image(self) -> Optional[ImageFrame]:
        return self.images[-1][1] if self.images else None

    @property
    def last_text(self) -> Optional[str]:
        return self.texts[-1][1] if self.texts else None

    @property
    def last_bytes(self) -> Optional[bytes]:
        return self.datas[-1][1] if self.datas else None

    def first_frame_latency_s(self) -> Optional[float]:
        """Seconds of received audio consumed before the first complete
        framed payload was published (sync acquisition + one full frame)."""
        events = self.images or self.texts or self.datas
        return events[0][0] / self.fs if events else None

    @property
    def startup_lag(self) -> int:
        """Leading run of exact-zero samples the decoder pads while its FIFOs
        and row batching fill (measured, not modelled)."""
        nonzero = np.flatnonzero(self.decoded_audio)
        return int(nonzero[0]) if nonzero.size else len(self.decoded_audio)

    @property
    def aligned_decoded_audio(self) -> np.ndarray:
        if self.expected_audio is None:
            return self.decoded_audio[self.startup_lag:]
        lag = self.startup_lag
        return self.decoded_audio[lag: lag + len(self.expected_audio)]

    def realtime_factor(self) -> float:
        """(encode+decode CPU time) / audio duration; < 1 means real-time."""
        return (self.encode_wall_s + self.decode_wall_s) * self.fs / max(len(self.encoded), 1)


# ── ground truth helpers ────────────────────────────────────────────────────

def _ground_truth_image(path: str, settings) -> ImageFrame:
    from PIL import Image
    w, h, ch = settings.image_target_w, settings.image_target_h, settings.image_channels
    with Image.open(path) as img:
        img = img.convert("L" if ch == 1 else "RGB").resize((w, h), Image.BILINEAR)
        return (img.tobytes(), w, h, ch)


def _expected_audio(payload: AudioPayload, settings, num_samples: int) -> np.ndarray:
    """What a perfect decoder would emit: the payload resampled to the symbol
    rate (MSG_FS), then stretched back to chunk length (same convention as
    exp/harness.py)."""
    raw = payload.get_data()
    if not raw:
        return np.zeros(num_samples)
    if payload.get_sample_rate() > 0:
        native, target = payload.get_sample_rate(), int(settings.MSG_FS)
        d = gcd(native, target)
        raw = resample_poly(np.array(raw, dtype=np.float32), target // d, native // d).tolist()
    out: List[float] = []
    i, n = 0, len(raw)
    while len(out) < num_samples:
        row = [raw[(i + j) % n] for j in range(settings.data_harmonics)]
        out += SymbolRow(row).resample_to_size(settings.chunk_size)
        i = (i + settings.data_harmonics) % n
    return np.asarray(out[:num_samples])


def render_carrier(config: ExperimentConfig, num_samples: Optional[int] = None) -> np.ndarray:
    """The same carrier with an EMPTY payload (no bits encoded) — reference
    signal for imperceptibility metrics (encoded vs clean carrier)."""
    settings = config.make_settings()
    from StrategyKinds import apply_strategy_kind
    apply_strategy_kind(settings, config.strategy_kind)
    settings.set_bits_per_symbol(config.bits_per_symbol)

    serializer = AudioSerializer(settings, SerializerMode.DIGITAL)
    strategy = ENCODING_STRATEGY_CLASSES[config.strategy_kind](
        settings, AdditiveWaveGenerator.harmonic(settings), serializer)
    strategy.load_payload(AudioPayload())
    encoder = Encoder(strategy)
    encoder.set_f0(config.f0)

    if num_samples is None:
        num_samples = int(config.duration_s * settings.fs_out)
    out: List[float] = []
    block = settings.audio_driver_polling_rate
    while len(out) < num_samples:
        out += encoder.process(block).get_samples()
    return np.asarray(out[:num_samples])


# ── the runner ──────────────────────────────────────────────────────────────

def run_experiment(config: ExperimentConfig) -> RunResult:
    """One full encode -> channel -> decode round trip, headless."""
    # Encoder and decoder each get their own Settings: both facades mutate
    # chunk_size on strategy switches, and a shared object would alias.
    enc_settings = config.make_settings()
    dec_settings = config.make_settings()

    enc = EncoderDSP(enc_settings)
    enc.set_strategy_kind(config.strategy_kind)
    enc.set_payload_kind(config.payload_kind)
    enc.set_codec_mode(config.codec_mode)
    enc.set_bits_per_symbol(config.bits_per_symbol)
    if config.payload_path is not None:
        enc.load_payload_file(config.payload_path)
    elif config.payload_kind in ("binary", "text"):
        raise ValueError(f"payload_path is required for kind '{config.payload_kind}'")
    enc.set_f0(config.f0)

    dec = DecoderDSP(dec_settings)
    dec.set_strategy_kind(config.strategy_kind)
    dec.set_payload_kind(config.payload_kind)
    dec.set_codec_mode(config.codec_mode)
    dec.set_bits_per_symbol(config.bits_per_symbol)
    from Sink import SinkBehaviour
    dec.set_sink_behaviour(
        SinkBehaviour.CLEAN if config.sink_behaviour == "clean" else SinkBehaviour.LIVE)
    dec.set_f0_estimator_mode(config.f0_mode)
    dec.set_pitch_quantize(config.pitch_quantize)
    dec.set_f0(config.decoder_f0 if config.decoder_f0 is not None else config.f0)

    settings = enc.settings
    fs = int(settings.fs_out)
    block = config.block_size or settings.audio_driver_polling_rate
    num_blocks = max(1, int(round(config.duration_s * fs / block)))

    # ── encode ──
    t0 = time.perf_counter()
    encoded_parts = [enc.process(block).get_samples() for _ in range(num_blocks)]
    encode_wall_s = time.perf_counter() - t0
    encoded = np.asarray([s for part in encoded_parts for s in part])

    # ── channel ──
    received = encoded if config.channel is None else np.asarray(config.channel(encoded, fs))

    # ── decode ──
    result_stub: dict = {"images": [], "raw_images": [], "texts": [], "datas": []}
    consumed = [0]  # input-sample index, captured by the callbacks below

    dec.set_on_image(lambda frame: result_stub["images"].append((consumed[0], frame)))
    dec.set_on_raw_image(lambda frame: result_stub["raw_images"].append((consumed[0], frame)))
    dec.set_on_text(lambda text: result_stub["texts"].append((consumed[0], text)))
    dec.set_on_data(lambda data: result_stub["datas"].append((consumed[0], data)))

    decoded_audio: List[float] = []
    f0_track: List[float] = []
    confidence_track: List[float] = []
    gated_blocks = 0

    t0 = time.perf_counter()
    for start in range(0, len(received), block):
        chunk = received[start:start + block]
        if len(chunk) < block:
            break
        consumed[0] = start + block
        decoded_audio += dec.process_chunk(chunk.tolist(), block)
        f0_track.append(dec.get_estimated_f0() if config.f0_mode != "manual" else config.f0)
        confidence_track.append(dec.get_confidence())
        gated_blocks += int(dec.is_gated())
    decode_wall_s = time.perf_counter() - t0
    decoded_audio_arr = np.asarray(decoded_audio)

    # ── ground truth ──
    gt_image = gt_text = gt_bytes = expected_audio = None
    if config.payload_kind == "image":
        gt_image = _ground_truth_image(enc.get_payload_path(), settings)
    elif config.payload_kind == "text":
        with open(config.payload_path, "r", encoding="utf-8") as f:
            gt_text = f.read()
    elif config.payload_kind == "binary":
        with open(config.payload_path, "rb") as f:
            gt_bytes = f.read()
    else:
        payload = AudioPayload()
        payload.load_from_file(enc.get_payload_path())
        nonzero = np.flatnonzero(decoded_audio_arr)
        lag = int(nonzero[0]) if nonzero.size else 0
        expected_audio = _expected_audio(payload, settings, len(decoded_audio_arr) - lag)

    return RunResult(
        config=config, fs=fs, chunk_size=settings.chunk_size,
        encoded=encoded, received=received, decoded_audio=decoded_audio_arr,
        images=result_stub["images"], raw_images=result_stub["raw_images"],
        texts=result_stub["texts"], datas=result_stub["datas"],
        ground_truth_image=gt_image, ground_truth_text=gt_text,
        ground_truth_bytes=gt_bytes, expected_audio=expected_audio,
        f0_track=f0_track, confidence_track=confidence_track, gated_blocks=gated_blocks,
        encode_wall_s=encode_wall_s, decode_wall_s=decode_wall_s,
    )
