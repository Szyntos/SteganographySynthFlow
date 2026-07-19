from typing import List

from AudioChunk import AudioChunk
from Payload import SymbolRow
from SamplesFifo import SamplesFifo
from Settings import Settings
from Sink import SymbolSink
from .DecodingStrategy import DecodingStrategy
from .RowToAudioResampler import ResampleMethod, RowToAudioResampler


class Decoder:
    def __init__(
            self,
            settings: Settings,
            decoding_strategy: DecodingStrategy,
            sink: SymbolSink,
            resample_method: "ResampleMethod | str" = ResampleMethod.POLY,
    ):
        self._settings = settings
        self._decoding_strategy: DecodingStrategy = decoding_strategy
        self._sink: SymbolSink = sink
        self._resampler = RowToAudioResampler(settings, resample_method)
        self._max_driver_block_size: int = 0
        self._audio_chunk_output_fifo: SamplesFifo = SamplesFifo()
        self.reconfigure()

    def set_resample_method(self, resample_method: "ResampleMethod | str") -> None:
        self._resampler.set_resample_method(resample_method)

    def get_resample_method(self) -> ResampleMethod:
        return self._resampler.get_resample_method()

    def reconfigure(self) -> None:
        self._max_driver_block_size = self._settings.max_driver_block_size
        self._audio_chunk_output_fifo = SamplesFifo(
            self._decoding_strategy.get_internal_clock() + self._max_driver_block_size - 1
        )
        self._resampler.reconfigure(self._decoding_strategy.get_internal_clock(), self._settings.data_harmonics)

    def get_output_fifo_size(self) -> int:
        return self._audio_chunk_output_fifo.get_size()

    def process(self, input_samples: AudioChunk, num_samples: int, gated: bool = False) -> AudioChunk:
        decoded_symbols: List[SymbolRow] = self._decoding_strategy.decode_samples(
            input_samples, num_samples, gated=gated)
        for symbol_row in decoded_symbols:
            samples = self._resampler.push_row(symbol_row.get_offsets())
            if samples:
                self._audio_chunk_output_fifo.push(samples)
        self._sink.push_many(decoded_symbols)
        return AudioChunk(self._audio_chunk_output_fifo.pop_or_silence(num_samples))
