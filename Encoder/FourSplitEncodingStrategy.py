from AdditiveWaveGenerator import AdditiveWaveGenerator
from Framing.SplitLayout import SplitLayout
from Serializer import Serializer
from Settings import Settings
from .EncodingStrategy import EncodingStrategy


class FourSplitEncodingStrategy(EncodingStrategy):
    """Splits each chunk into 4 quarters: pilot, ramp-up, data, ramp-down.

    Quarter 0 carries no phase offset (pilot). Quarter 1 linearly ramps the
    phase offset envelope from 0 to 1 (pilot -> data). Quarter 2 holds the
    full phase offset (data). Quarter 3 linearly ramps the envelope back
    from 1 to 0 (data -> pilot), so the chunk begins and ends on the pilot
    phase, matching the C++ EncodeSplit4 envelope.
    """

    def __init__(self, settings: Settings, additive_wave_generator: AdditiveWaveGenerator, serializer: Serializer):
        super().__init__(settings, additive_wave_generator, serializer)

    def reconfigure(self) -> None:
        super().reconfigure()
        self._layout = SplitLayout.four_split(self._internal_clock)
