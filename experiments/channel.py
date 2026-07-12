"""Channel impairment models.

Every impairment is a callable ``(samples: np.ndarray, fs: int) -> np.ndarray``
applied to the full encoded signal between encoder and decoder. Compose them
with ``Chain``. All are deterministic given ``seed`` so runs are repeatable.

These model what the thesis cares about: what survives an acoustic or lossy
path? Noise, gain, filtering, clipping, quantization, dropped audio blocks,
sample-clock offset, pitch/f0 mismatch (via resampling), and reverb.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as sps


@dataclass
class Chain:
    """Apply impairments in order."""
    stages: Tuple = ()

    def __init__(self, *stages):
        self.stages = tuple(stages)

    def __call__(self, x: np.ndarray, fs: int) -> np.ndarray:
        for stage in self.stages:
            x = stage(x, fs)
        return x

    def __repr__(self) -> str:
        return " -> ".join(repr(s) for s in self.stages) or "Identity"


@dataclass
class AWGN:
    """Additive white Gaussian noise at a target SNR relative to the signal."""
    snr_db: float
    seed: int = 0

    def __call__(self, x: np.ndarray, fs: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        p_sig = float(np.mean(np.square(x)))
        if p_sig <= 0.0:
            return x
        p_noise = p_sig / (10.0 ** (self.snr_db / 10.0))
        return x + rng.normal(0.0, np.sqrt(p_noise), size=x.shape)


@dataclass
class Gain:
    gain_db: float

    def __call__(self, x: np.ndarray, fs: int) -> np.ndarray:
        return x * (10.0 ** (self.gain_db / 20.0))


@dataclass
class HardClip:
    """Clip at a fraction of the signal's own peak (models amp/mic overload)."""
    clip_at_peak_fraction: float = 0.5

    def __call__(self, x: np.ndarray, fs: int) -> np.ndarray:
        limit = float(np.max(np.abs(x))) * self.clip_at_peak_fraction
        return np.clip(x, -limit, limit) if limit > 0 else x


@dataclass
class Quantize:
    """Round to a fixed bit depth (models cheap ADC / low-bit transport)."""
    bits: int = 8

    def __call__(self, x: np.ndarray, fs: int) -> np.ndarray:
        peak = float(np.max(np.abs(x))) or 1.0
        levels = 2 ** (self.bits - 1)
        return np.round(x / peak * levels) / levels * peak


@dataclass
class Butterworth:
    """Band-limiting filter (models speaker/mic/telephone band)."""
    btype: str = "lowpass"     # "lowpass" | "highpass" | "bandpass"
    cutoff_hz: Tuple[float, ...] = (16_000.0,)
    order: int = 4

    def __call__(self, x: np.ndarray, fs: int) -> np.ndarray:
        wn = [c / (fs / 2.0) for c in self.cutoff_hz]
        sos = sps.butter(self.order, wn if len(wn) > 1 else wn[0],
                         btype=self.btype, output="sos")
        return sps.sosfilt(sos, x)


@dataclass
class DropBlocks:
    """Zero out random spans of ``block_len`` samples with probability
    ``drop_prob`` each — exercises EnergyGate + DropTolerance recovery."""
    drop_prob: float = 0.05
    block_len: int = 960
    seed: int = 0

    def __call__(self, x: np.ndarray, fs: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        y = x.copy()
        for start in range(0, len(y), self.block_len):
            if rng.random() < self.drop_prob:
                y[start:start + self.block_len] = 0.0
        return y


@dataclass
class SampleShift:
    """Delete (or prepend) ``shift`` samples: mis-alignment between encoder
    chunk boundaries and the decoder FIFO — what the GUI 'tune' slider fixes."""
    shift: int = 0

    def __call__(self, x: np.ndarray, fs: int) -> np.ndarray:
        if self.shift >= 0:
            return x[self.shift:]
        return np.concatenate([np.zeros(-self.shift), x])


@dataclass
class ClockSkew:
    """Resample by ``ppm`` parts-per-million (sample-clock mismatch between
    two sound cards). Also usable coarsely as a pitch-shift impairment."""
    ppm: float = 100.0

    def __call__(self, x: np.ndarray, fs: int) -> np.ndarray:
        ratio = 1.0 + self.ppm * 1e-6
        n_out = int(round(len(x) * ratio))
        return sps.resample(x, n_out)


@dataclass
class Echo:
    """Single reflection: crude room model."""
    delay_ms: float = 30.0
    attenuation: float = 0.3

    def __call__(self, x: np.ndarray, fs: int) -> np.ndarray:
        d = int(fs * self.delay_ms / 1000.0)
        y = x.copy()
        if 0 < d < len(x):
            y[d:] += self.attenuation * x[:-d]
        return y


@dataclass
class Reverb:
    """Exponentially decaying noise impulse response (diffuse room tail)."""
    rt60_ms: float = 200.0
    wet: float = 0.2
    seed: int = 0

    def __call__(self, x: np.ndarray, fs: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n = int(fs * self.rt60_ms / 1000.0)
        t = np.arange(n) / fs
        ir = rng.normal(size=n) * np.exp(-6.91 * t / (self.rt60_ms / 1000.0))
        ir /= np.max(np.abs(np.convolve(np.ones(10), ir))) + 1e-12
        tail = sps.fftconvolve(x, ir)[: len(x)]
        return (1.0 - self.wet) * x + self.wet * tail
