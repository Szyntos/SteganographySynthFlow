from Framing.SplitLayout import SplitLayout
from Settings import Settings
from .DecodingStrategy import DecodingStrategy


class FourSplitDecodingStrategy(DecodingStrategy):
    """Decodes chunks produced by FourSplitEncodingStrategy.

    Only the stable pilot quarter (samples [0, q)) and the stable data
    quarter (samples [2q, 3q)) are analyzed; the ramp quarters in between
    are skipped since their envelope is in transition and would bias the
    phase estimate, matching the C++ DecodeSplit4 projection.
    """

    def __init__(self, settings: Settings):
        super().__init__(settings)

    def reconfigure(self) -> None:
        super().reconfigure()
        self._layout = SplitLayout.four_split(self._internal_clock)
