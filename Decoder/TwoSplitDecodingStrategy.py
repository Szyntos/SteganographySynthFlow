from Framing.SplitLayout import SplitLayout
from Settings import Settings
from .DecodingStrategy import DecodingStrategy


class TwoSplitDecodingStrategy(DecodingStrategy):
    def __init__(self, settings: Settings):
        super().__init__(settings)

    def reconfigure(self) -> None:
        super().reconfigure()
        self._layout = SplitLayout.two_split(self._internal_clock)
