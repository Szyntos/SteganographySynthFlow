import math
from dataclasses import dataclass

import numpy as np


@dataclass
class EnergyGateConfig:
    ema_alpha: float = 0.05
    abs_floor: float = 1e-6
    drop_ratio: float = 0.25


class EnergyGate:
    """Adaptive silence/weak-signal gate, ported from aoa_cpp_2's runtime::EnergyGate.

    Tracks a slow EMA of the chunk RMS and drops a chunk if its RMS falls
    below an absolute floor, or well below the recent EMA (drop_ratio).
    """

    def __init__(self, cfg: EnergyGateConfig = None):
        self._cfg = cfg if cfg is not None else EnergyGateConfig()
        self._ema_valid = False
        self._ema = 0.0

    def reset(self) -> None:
        self._ema_valid = False
        self._ema = 0.0

    @staticmethod
    def rms(samples: np.ndarray) -> float:
        if len(samples) == 0:
            return 0.0
        x = np.asarray(samples, dtype=np.float64)
        mean = float(np.mean(x * x))
        return math.sqrt(mean + 1e-18)

    def is_drop(self, rms_now: float) -> bool:
        cfg = self._cfg
        if not self._ema_valid:
            self._ema = rms_now
            self._ema_valid = True
            return rms_now < cfg.abs_floor

        self._ema = (1.0 - cfg.ema_alpha) * self._ema + cfg.ema_alpha * rms_now
        if rms_now < cfg.abs_floor:
            return True
        if self._ema > 0.0 and rms_now < self._ema * cfg.drop_ratio:
            return True
        return False

    @property
    def ema(self) -> float:
        return self._ema

    @property
    def ema_valid(self) -> bool:
        return self._ema_valid
