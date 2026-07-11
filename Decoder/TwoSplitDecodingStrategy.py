import cmath
import math
from typing import List

import numpy as np

from AdditiveWaveGenerator import AdditiveWaveGenerator
from Payload import SymbolRow
from Settings import Settings
from .DecodingStrategy import DecodingStrategy


class TwoSplitDecodingStrategy(DecodingStrategy):
    def __init__(self, settings: Settings, additive_wave_generator: AdditiveWaveGenerator):
        self._m_state: np.ndarray = np.zeros(0)
        self._alpha: float = settings.decoder_strategy_alpha
        self._mag_threshold: float = 1e-6
        self._analysis_cache_f0: float | None = None
        self._analysis_cache: tuple | None = None
        super().__init__(settings, additive_wave_generator)

    def reconfigure(self) -> None:
        super().reconfigure()
        self._m_state = np.zeros(self._num_rows)
        self._analysis_cache_f0 = None
        self._analysis_cache = None

    def _get_analysis_arrays(self, fs: float, half: int):
        if self._analysis_cache_f0 == self._f0 and self._analysis_cache is not None:
            return self._analysis_cache

        total_harmonics = self._settings.total_harmonics
        nyquist = 0.5 * fs

        # Use the largest integer-cycle window length that fits in half.
        # This ensures harmonics are mutually orthogonal, eliminating cross-harmonic
        # leakage that otherwise biases the phase measurement when f0*half/fs is
        # non-integer (e.g. f0=500, half=240, fs=48000 → 2.5 cycles, not integer).
        cycles_per_period = max(1, int(round(fs / self._f0)))
        n_cycles = half // cycles_per_period
        if n_cycles < 1:
            # f0 too low to fit even one cycle in half; use all available samples
            analysis_len = half
        else:
            analysis_len = n_cycles * cycles_per_period
        win_start = (half - analysis_len) // 2  # centre window in each half

        n = np.arange(analysis_len, dtype=np.float64)
        hann_win = 0.5 - 0.5 * np.cos(2.0 * math.pi * n / (analysis_len - 1)) if analysis_len > 1 else np.ones(1)

        t_p = n + win_start
        t_d = n + half + win_start

        # omega_k = 2*pi*(k+1)*f0/fs for k in [0..total_harmonics-1]
        harmonic_nums = np.arange(1, total_harmonics + 1, dtype=np.float64)
        freqs = harmonic_nums * self._f0
        valid_mask = (freqs > 0.0) & (freqs <= nyquist)
        omegas = np.where(valid_mask, 2.0 * math.pi * freqs / fs, 0.0)

        # Precompute the DFT projection matrices; these only depend on f0/fs/half.
        phase_p = np.outer(omegas, t_p)
        phase_d = np.outer(omegas, t_d)
        proj_p = np.exp(-1j * phase_p)
        proj_d = np.exp(-1j * phase_d)

        self._analysis_cache_f0 = self._f0
        self._analysis_cache = (analysis_len, win_start, hann_win, valid_mask, proj_p, proj_d)
        return self._analysis_cache

    def _decode(self, samples: List[float]) -> SymbolRow:
        if self._f0 <= 0.0:
            return SymbolRow(self._m_state.tolist())

        fs = float(self._settings.fs_out)
        data_offset = self._settings.data_offset
        data_harmonics = self._settings.data_harmonics
        phase_range = float(self._settings.phase_range)
        total_harmonics = self._settings.total_harmonics
        half = self._internal_clock // 2

        analysis_len, win_start, hann_win, valid_mask, proj_p, proj_d = self._get_analysis_arrays(fs, half)

        samples_arr = np.asarray(samples, dtype=np.float64)
        xp = samples_arr[win_start:win_start + analysis_len] * hann_win
        xd = samples_arr[half + win_start:half + win_start + analysis_len] * hann_win

        # Vectorised DFT projections: (H, analysis_len) outer product, precomputed above
        Z_p = (xp * proj_p).sum(axis=1)
        Z_d = (xd * proj_d).sum(axis=1)
        Z_p = np.where(valid_mask, Z_p, 0j)
        Z_d = np.where(valid_mask, Z_d, 0j)

        ap = np.abs(Z_p)
        ad = np.abs(Z_d)

        # Bias phase from non-data-band harmonics
        non_data = np.ones(total_harmonics, dtype=bool)
        non_data[data_offset:data_offset + data_harmonics] = False
        bias_mask = valid_mask & non_data & (ap >= self._mag_threshold) & (ad >= self._mag_threshold)

        sum_unit = complex(0.0, 0.0)
        if bias_mask.any():
            prod_bias = Z_d[bias_mask] * np.conj(Z_p[bias_mask])
            w = ap[bias_mask] * ad[bias_mask]
            sum_unit = np.sum(w * np.exp(1j * np.angle(prod_bias)))

        phi_bias = cmath.phase(sum_unit) if abs(sum_unit) > 0.0 else 0.0

        # Decode each data harmonic
        n_data = min(data_harmonics, total_harmonics - data_offset)
        h_idx = np.arange(data_offset, data_offset + n_data)

        data_valid = valid_mask[h_idx] & (ap[h_idx] >= self._mag_threshold) & (ad[h_idx] >= self._mag_threshold)

        prod_data = Z_d[h_idx] * np.conj(Z_p[h_idx])
        delta = np.angle(prod_data)
        delta_centered = (delta - phi_bias + math.pi) % (2.0 * math.pi) - math.pi
        m_hat = delta_centered / phase_range

        out = np.where(
            data_valid,
            (1.0 - self._alpha) * self._m_state[:n_data] + self._alpha * m_hat,
            self._m_state[:n_data],
        )

        self._m_state[:n_data] = out

        # Pad to data_harmonics if total_harmonics - data_offset < data_harmonics
        if n_data < data_harmonics:
            out = np.concatenate([out, self._m_state[n_data:data_harmonics]])

        return SymbolRow(out.tolist())
