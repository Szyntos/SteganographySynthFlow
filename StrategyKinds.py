from Encoder import FourSplitEncodingStrategy, TwoSplitEncodingStrategy
from Decoder import FourSplitDecodingStrategy, TwoSplitDecodingStrategy

# The 4-split strategy's decode window (chunk_size/4) is half as long as the
# 2-split strategy's (chunk_size/2), so it needs double the chunk_size to
# keep the same DFT bin spacing between harmonics and avoid leaking Hann
# window energy across neighboring harmonics. See EncodingStrategy/
# DecodingStrategy classes.
ENCODING_STRATEGY_CLASSES = {
    "two": TwoSplitEncodingStrategy,
    "four": FourSplitEncodingStrategy,
}

DECODING_STRATEGY_CLASSES = {
    "two": TwoSplitDecodingStrategy,
    "four": FourSplitDecodingStrategy,
}


def scaled_chunk_size(settings, base_chunk_size: int, strategy_kind: str) -> int:
    return base_chunk_size * settings.strategy_chunk_size_multiplier[strategy_kind]
