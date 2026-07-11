"""
Dumps six wav files to exp/output:
1. encoded.wav        - full AudioPayload run through the real encoder
2. carrier_inv.wav    - same sawtooth carrier with an empty payload (no bits
                        encoded), phase-inverted
3. saw_diff.wav       - encoded.wav + carrier_inv.wav (sample-wise)
4. to_encode.wav      - the input AudioPayload after the serializer resamples
                        it down to MSG_FS (one symbol-row sample per data
                        harmonic slot) - what actually gets encoded
5. decoded.wav        - to_encode.wav's data recovered by running the
                        encoded signal back through the real decoder
6. data_diff.wav      - to_encode.wav - decoded.wav (sample-wise, after
                        downsampling decoded back to MSG_FS)
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Decoder import Decoder, TwoSplitDecodingStrategy
from Encoder import Encoder, TwoSplitEncodingStrategy
from Framing import FramingSyncController
from Payload import AudioPayload
from Serializer import AudioSerializer
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import AudioSink, SinkBehaviour

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
F0 = 400.0


def harmonic_generator(settings: Settings) -> AdditiveWaveGenerator:
    gen = AdditiveWaveGenerator(settings)
    gen.set_omegas([float(i + 1) for i in range(settings.total_harmonics)])
    gen.set_phases([0.0] * settings.total_harmonics)
    gen.set_amps([settings.base_amplitude / (i + 1) for i in range(settings.total_harmonics)])
    return gen


def build_encoder(settings: Settings, payload: AudioPayload) -> tuple[Encoder, AudioSerializer]:
    serializer = AudioSerializer(settings, SerializerMode.DIGITAL)
    strategy = TwoSplitEncodingStrategy(settings, harmonic_generator(settings), serializer)
    strategy.load_payload(payload)
    encoder = Encoder(strategy)
    encoder.set_f0(F0)
    return encoder, serializer


def chunks_for_one_full_loop(serializer: AudioSerializer, settings: Settings) -> int:
    """Number of chunks needed so the serializer consumes every symbol in the
    payload exactly once (each chunk pulls settings.data_harmonics symbols)."""
    size = serializer._serialized_payload.get_size()
    if size == 0:
        return 1
    return -(-size // settings.data_harmonics)  # ceil division


def render(encoder: Encoder, settings: Settings, num_chunks: int) -> np.ndarray:
    samples = []
    for _ in range(num_chunks):
        samples += encoder.process(settings.chunk_size).get_samples()
    return np.array(samples, dtype=np.float32)


def build_decoder(settings: Settings) -> tuple[Decoder, TwoSplitDecodingStrategy]:
    decoding_strategy = TwoSplitDecodingStrategy(settings, harmonic_generator(settings))
    decoding_strategy.set_f0(F0)
    sink = AudioSink(FramingSyncController(), SinkBehaviour.LIVE, settings=settings)
    decoder = Decoder(settings, decoding_strategy, sink)
    return decoder, decoding_strategy


def decode(encoded: np.ndarray, settings: Settings, num_chunks: int) -> np.ndarray:
    decoder, _ = build_decoder(settings)
    encoded_list = encoded.tolist()
    samples = []
    for i in range(num_chunks):
        chunk = encoded_list[i * settings.chunk_size:(i + 1) * settings.chunk_size]
        samples += decoder.process(AudioChunk(chunk), settings.chunk_size).get_samples()
    return np.array(samples, dtype=np.float32)


def measure_startup_lag(decoded: np.ndarray) -> int:
    """Decoder.process pads missing output with literal 0.0 (SamplesFifo.
    pop_or_silence) until TwoSplitDecodingStrategy's row-batching lookahead
    (decoder_batch_rows + decoder_lookahead_rows rows) has filled - a longer,
    different delay than a generic FIFO-fill formula predicts. Measure it
    directly as the leading run of exact-zero samples rather than guessing."""
    nonzero = np.flatnonzero(decoded)
    return int(nonzero[0]) if nonzero.size else 0


def normalize(*arrays: np.ndarray, headroom: float = 0.95) -> None:
    """In-place peak-normalize a group of arrays together, preserving their
    relative amplitudes, so none of them clip in the wav output."""
    peak = max((np.max(np.abs(a)) if a.size else 0.0) for a in arrays)
    peak = max(peak, 1e-9)
    scale = headroom / peak
    for a in arrays:
        a *= scale


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    settings = Settings()
    settings.validate()

    modulator_path = settings.modulator_wav_path
    if not os.path.isabs(modulator_path):
        modulator_path = os.path.join(REPO_ROOT, modulator_path)

    real_payload = AudioPayload()
    real_payload.load_from_file(modulator_path)
    real_encoder, real_serializer = build_encoder(settings, real_payload)
    num_chunks = chunks_for_one_full_loop(real_serializer, settings)
    encoded = render(real_encoder, settings, num_chunks)

    # What the serializer actually feeds the encoder: the input payload
    # resampled down to one sample per data-harmonic slot (MSG_FS rate).
    to_encode = np.array(real_serializer._serialized_payload.get_offsets(), dtype=np.float32)

    # Round-trip the encoded signal back through the real decoder to recover
    # to_encode.wav's data.
    decoded = decode(encoded, settings, num_chunks)

    # The decoder's output lags its input by a startup delay (see
    # measure_startup_lag); drop that lead-in so decoded[0] lines up with
    # to_encode[0].
    startup_lag = measure_startup_lag(decoded)
    decoded = decoded[startup_lag:]

    empty_payload = AudioPayload()
    empty_encoder, _ = build_encoder(settings, empty_payload)
    carrier = render(empty_encoder, settings, num_chunks)
    carrier_inv = -carrier

    saw_diff = encoded + carrier_inv

    # decoded is at audio rate (chunk_size samples/row); bring it back down to
    # to_encode's data rate (one sample/row) before diffing them directly.
    decoded_at_data_rate = resample_poly(decoded, settings.data_harmonics, settings.chunk_size)
    n = min(len(to_encode), len(decoded_at_data_rate))
    data_diff = to_encode[:n] - decoded_at_data_rate[:n]

    # Summing base_amplitude/n across total_harmonics harmonics can peak well
    # past +/-1 (worst case ~ base_amplitude * sum(1/n)), so scale each group
    # of buffers together (preserving relative amplitudes) to avoid clipping.
    # encoded/carrier/saw_diff share the audio-domain scale; to_encode/decoded/
    # data_diff are a separate (data-domain) scale, so they're normalized
    # independently.
    normalize(encoded, carrier_inv, saw_diff)
    normalize(to_encode, decoded, data_diff)

    # The encoder/carrier/diff waveforms are generated at settings.fs_out (the
    # phase deltas in AdditiveWaveGenerator use fs_out); to_encode/decoded live
    # in the data domain at settings.MSG_FS (one sample per data-harmonic
    # slot, resampled up to chunk_size by the decoder - see Sink/AudioSink.py
    # for the same MSG_FS convention).
    audio_rate = int(settings.fs_out)
    data_rate = int(settings.MSG_FS)
    sf.write(os.path.join(OUTPUT_DIR, "encoded.wav"), encoded, audio_rate, format="WAV", subtype="FLOAT")
    sf.write(os.path.join(OUTPUT_DIR, "carrier_inv.wav"), carrier_inv, audio_rate, format="WAV", subtype="FLOAT")
    sf.write(os.path.join(OUTPUT_DIR, "saw_diff.wav"), saw_diff, audio_rate, format="WAV", subtype="FLOAT")
    sf.write(os.path.join(OUTPUT_DIR, "to_encode.wav"), to_encode, data_rate, format="WAV", subtype="FLOAT")
    sf.write(os.path.join(OUTPUT_DIR, "decoded.wav"), decoded, audio_rate, format="WAV", subtype="FLOAT")
    sf.write(os.path.join(OUTPUT_DIR, "data_diff.wav"), data_diff, data_rate, format="WAV", subtype="FLOAT")

    print(f"Wrote {len(encoded)} samples ({num_chunks} chunks, one full serializer loop) "
          f"at {audio_rate} Hz to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
