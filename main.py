import math
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from AdditiveWaveGenerator import AdditiveWaveGenerator
from Decoder import Decoder, DecodingStrategy, TwoSplitDecodingStrategy
from Deserializer import Deserializer, AudioDeserializer
from Encoder import Encoder, EncodingStrategy, TwoSplitEncodingStrategy
from Framing import FramingSyncController
from MidiInput import MidiInput
from Payload import AudioPayload, Payload, SymbolRow
from Serializer import AudioSerializer, Serializer
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import SinkBehaviour, AudioSink, Sink

settings: Settings = Settings()

def harmonic_generator(settings: Settings) -> AdditiveWaveGenerator:
    additive_wave_generator: AdditiveWaveGenerator = AdditiveWaveGenerator(settings)

    additive_wave_generator.set_omegas(
        [float(i + 1) for i in range(settings.total_harmonics)]
    )
    additive_wave_generator.set_phases(
        [0.0] * settings.total_harmonics
    )
    additive_wave_generator.set_amps(
        [settings.base_amplitude / (i + 1) for i in range(settings.total_harmonics)]
    )
    return additive_wave_generator

def compute_startup_lag(settings: Settings, num_samples: int) -> int:
    chunk_size = settings.chunk_size
    startup_threshold = chunk_size + settings.max_driver_block_size - 1
    input_fifo = 0
    output_fifo = 0
    silence = 0
    while True:
        input_fifo += num_samples
        while input_fifo >= chunk_size:
            input_fifo -= chunk_size
            output_fifo += chunk_size
        if output_fifo >= startup_threshold:
            break
        silence += num_samples
    return silence


def get_expected_decoded_signal(payload: AudioPayload, settings: Settings, num_samples_needed: int):
    raw = payload.get_data()
    if payload.get_sample_rate() > 0:
        native_rate = payload.get_sample_rate()
        target_rate = int(settings.MSG_FS)
        divisor = gcd(native_rate, target_rate)
        up = target_rate // divisor
        down = native_rate // divisor
        raw = resample_poly(np.array(raw, dtype=np.float32), up, down).tolist()

    data_harmonics = settings.data_harmonics
    chunk_size = settings.chunk_size
    n = len(raw)
    result = []
    i = 0
    while len(result) < num_samples_needed:
        chunk = [raw[(i + j) % n] for j in range(data_harmonics)]
        result += SymbolRow(chunk).resample_to_size(chunk_size)
        i = (i + data_harmonics) % n
    return result[:num_samples_needed]


def graph_encoded_and_decoded(encoded, decoded, expected, startup_lag):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(encoded)
    ax1.set_title("Encoded")
    ax1.set_ylabel("Amplitude")

    ax2.plot(decoded, label="Decoded")
    ax2.plot(range(startup_lag, startup_lag + len(expected)), expected, label="Expected", alpha=0.7)
    ax2.axvline(x=startup_lag, color='r', linestyle='--', alpha=0.5, label=f"Startup lag ({startup_lag} samples)")
    ax2.set_title("Decoded")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlabel("Sample index")
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    midi_input: MidiInput = MidiInput()

    payload: Payload = AudioPayload()
    payload.load_from_file(r"assets/sinesweep.wav")

    serializer: Serializer = AudioSerializer(settings, SerializerMode.DIGITAL)

    encoding_strategy: EncodingStrategy = TwoSplitEncodingStrategy(
        settings, harmonic_generator(settings), serializer
    )
    encoding_strategy.load_payload(payload)

    encoder: Encoder = Encoder(encoding_strategy)

    decoding_strategy: DecodingStrategy = TwoSplitDecodingStrategy(settings, harmonic_generator(settings))

    framing_sync_controller: FramingSyncController = FramingSyncController()
    sink: Sink = AudioSink(framing_sync_controller, SinkBehaviour.LIVE)

    deserializer: Deserializer = AudioDeserializer(settings, sink, SerializerMode.DIGITAL)
    decoder: Decoder = Decoder(settings, decoding_strategy, deserializer)

    def on_trigger(f0: float) -> None:
        encoder.set_f0(f0)
        decoding_strategy.set_f0(f0)

    midi_input.on_play(on_trigger)
    midi_input.trigger(500.0)

    num_samples = settings.audio_driver_polling_rate
    encoded = []
    decoded = []
    for i in range(15):
        encoded_frames = encoder.process(num_samples)
        encoded += encoded_frames.get_samples()
        decoded_frames = decoder.process(encoded_frames, num_samples)
        decoded += decoded_frames.get_samples()
    startup_lag = compute_startup_lag(settings, num_samples)
    expected = get_expected_decoded_signal(payload, settings, len(decoded) - startup_lag)
    graph_encoded_and_decoded(encoded, decoded, expected, startup_lag)

if __name__ == '__main__':
    main()
