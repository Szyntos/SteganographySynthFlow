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
        self._alpha: float = 1.0
        self._mag_threshold: float = 1e-6
        self._hann_win: np.ndarray = np.zeros(0)
        self._t_p: np.ndarray = np.zeros(0)
        self._t_d: np.ndarray = np.zeros(0)
        super().__init__(settings, additive_wave_generator)

    def reconfigure(self) -> None:
        super().reconfigure()
        half = self._internal_clock // 2
        self._m_state = np.zeros(self._num_rows)
        n = np.arange(half, dtype=np.float64)
        self._hann_win = 0.5 - 0.5 * np.cos(2.0 * math.pi * n / (half - 1)) if half > 1 else np.ones(half)
        self._t_p = n
        self._t_d = n + half

    def _decode(self, samples: List[float]) -> SymbolRow:
        if self._f0 <= 0.0:
            return SymbolRow(self._m_state.tolist())

        fs = float(self._settings.fs_out)
        total_harmonics = self._settings.total_harmonics
        data_offset = self._settings.data_offset
        data_harmonics = self._settings.data_harmonics
        phase_range = float(self._settings.phase_range)
        nyquist = 0.5 * fs
        half = self._internal_clock // 2

        samples_arr = np.asarray(samples, dtype=np.float64)
        xp = samples_arr[:half] * self._hann_win
        xd = samples_arr[half:half * 2] * self._hann_win

        # omega_k = 2*pi*(k+1)*f0/fs for k in [0..total_harmonics-1]
        harmonic_nums = np.arange(1, total_harmonics + 1, dtype=np.float64)
        freqs = harmonic_nums * self._f0
        valid_mask = (freqs > 0.0) & (freqs <= nyquist)
        omegas = np.where(valid_mask, 2.0 * math.pi * freqs / fs, 0.0)

        # Vectorised DFT projections: (H, half) outer product
        phase_p = np.outer(omegas, self._t_p)
        phase_d = np.outer(omegas, self._t_d)
        Z_p = (xp * np.exp(-1j * phase_p)).sum(axis=1)
        Z_d = (xd * np.exp(-1j * phase_d)).sum(axis=1)
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
