from synthflow.core.AdditiveWaveGenerator import AdditiveWaveGenerator
from synthflow.Framing.SplitLayout import SplitLayout
from synthflow.Serializer import Serializer
from synthflow.core.Settings import Settings
from .EncodingStrategy import EncodingStrategy


class TwoSplitEncodingStrategy(EncodingStrategy):
    def __init__(self, settings: Settings, additive_wave_generator: AdditiveWaveGenerator, serializer: Serializer):
        super().__init__(settings, additive_wave_generator, serializer)

    def reconfigure(self) -> None:
        super().reconfigure()
        self._layout = SplitLayout.two_split(self._internal_clock)
