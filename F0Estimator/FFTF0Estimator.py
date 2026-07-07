import math

import numpy as np

from .F0EstimatorBase import F0Estimator


class FFTF0Estimator(F0Estimator):
    """Windowed-FFT peak-bin f0 estimator.

    Ported from aoa_cpp_2's f0_estimator_fft: Hann window, zero-padded FFT,
    strongest magnitude bin within [f_min_hz, f_max_hz] wins (no
    parabolic/quadratic bin refinement).
    """

    def __init__(
        self,
        n_fft: int = 4096,
        f_min_hz: float = 50.0,
        f_max_hz: float = 2000.0,
        rms_floor: float = 1e-6,
        use_pilot_half: bool = False,
    ):
        if n_fft <= 0 or (n_fft & (n_fft - 1)) != 0:
            raise ValueError("n_fft must be a power of 2")
        self._n_fft = n_fft
        self._f_min_hz = f_min_hz
        self._f_max_hz = f_max_hz
        self._rms_floor = rms_floor
        self._use_pilot_half = use_pilot_half

    def estimate(self, samples: np.ndarray, fs: float) -> float:
        n = len(samples)
        length = n // 2 if self._use_pilot_half else n
        if length < 2:
            return 0.0

        x = np.asarray(samples[:length], dtype=np.float64)

        rms = math.sqrt(float(np.mean(x * x)))
        if rms < self._rms_floor:
            return 0.0

        x = x - np.mean(x)

        idx = np.arange(length, dtype=np.float64)
        window = (
            0.5 - 0.5 * np.cos(2.0 * math.pi * idx / (length - 1))
            if length > 1 else np.ones(1)
        )
        windowed = x * window

        n_fft = self._n_fft
        buf = np.zeros(n_fft, dtype=np.float64)
        take = min(length, n_fft)
        buf[:take] = windowed[:take]

        mags = np.abs(np.fft.rfft(buf))
        half = n_fft // 2

        k_min = max(1, int(math.ceil(self._f_min_hz * n_fft / fs)))
        k_max = min(half - 1, int(math.floor(self._f_max_hz * n_fft / fs)))
        if k_max <= k_min or k_min >= half:
            return 0.0

        window_slice = mags[k_min:k_max + 1]
        best_k = k_min + int(np.argmax(window_slice))
        best_mag = mags[best_k]
        if not (best_mag > 0.0):
            return 0.0

        f0 = best_k * fs / n_fft
        if not math.isfinite(f0) or f0 < self._f_min_hz or f0 > self._f_max_hz:
            return 0.0
        return f0
