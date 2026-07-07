import math

import numpy as np

from .F0EstimatorBase import F0Estimator


class AutocorrF0Estimator(F0Estimator):
    """Plain time-domain normalized autocorrelation f0 estimator.

    Ported from aoa_cpp_2's f0_estimator_autocorr: brute-force search over
    all lags in [fs/f_max, fs/f_min], no parabolic interpolation.
    """

    def __init__(
        self,
        f_min_hz: float = 200.0,
        f_max_hz: float = 1200.0,
        rms_floor: float = 1e-4,
        use_pilot_half: bool = True,
        corr_threshold: float = 0.2,
    ):
        self._f_min_hz = f_min_hz
        self._f_max_hz = f_max_hz
        self._rms_floor = rms_floor
        self._use_pilot_half = use_pilot_half
        self._corr_threshold = corr_threshold

    def estimate(self, samples: np.ndarray, fs: float) -> float:
        n = len(samples)
        length = n // 2 if self._use_pilot_half else n
        if length < 16:
            return 0.0

        x = np.asarray(samples[:length], dtype=np.float64)

        rms = math.sqrt(float(np.mean(x * x)))
        if rms < self._rms_floor:
            return 0.0

        fmin = max(1.0, self._f_min_hz)
        fmax = max(fmin, self._f_max_hz)

        lag_min = max(1, int(math.floor(fs / fmax)))
        lag_max = min(length - 2, int(math.ceil(fs / fmin)))
        if lag_min >= lag_max:
            return 0.0

        x0 = x - np.mean(x)
        e0 = float(np.sum(x0 * x0))
        if e0 <= 1e-18:
            return 0.0

        best_r = -1.0
        best_lag = -1
        for lag in range(lag_min, lag_max + 1):
            m = length - lag
            a = x0[:m]
            b = x0[lag:lag + m]
            num = float(np.dot(a, b))
            e1 = float(np.dot(b, b))
            r = num / (math.sqrt(e0 * (e1 + 1e-18)) + 1e-18)
            if r > best_r:
                best_r = r
                best_lag = lag

        if best_r <= self._corr_threshold or best_lag <= 0:
            return 0.0

        f0 = fs / best_lag
        if not math.isfinite(f0) or f0 < fmin or f0 > fmax:
            return 0.0
        return f0
