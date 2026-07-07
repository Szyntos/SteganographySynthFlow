from abc import ABC, abstractmethod

import numpy as np


class F0Estimator(ABC):
    """Estimates the fundamental frequency (f0) of an audio chunk, in Hz.

    Implementations return 0.0 when no reliable f0 could be estimated.
    """

    def reset(self) -> None:
        pass

    @abstractmethod
    def estimate(self, samples: np.ndarray, fs: float) -> float:
        ...
