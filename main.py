import math

from AdditiveWaveGenerator import AdditiveWaveGenerator
from Decoder import Decoder, DecodingStrategy, TwoSplitDecodingStrategy
from Deserializer import Deserializer, AudioDeserializer
from Encoder import Encoder, EncodingStrategy, TwoSplitEncodingStrategy
from Framing import FramingSyncController
from MidiInput import MidiInput
from Payload import AudioPayload, Payload
from Serializer import AudioSerializer, Serializer
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import SinkBehaviour, AudioSink, Sink

settings: Settings = Settings()

def sawtooth(settings: Settings) -> AdditiveWaveGenerator:
    additive_wave_generator: AdditiveWaveGenerator = AdditiveWaveGenerator(settings)

    additive_wave_generator.set_omegas(
        [2.0 * math.pi * (i + 1) for i in range(settings.total_harmonics)]
    )
    additive_wave_generator.set_phases(
        [0.0] * settings.total_harmonics
    )
    additive_wave_generator.set_amps(
        [(2.0 / math.pi) * ((-1) ** i) / (i + 1) for i in range(settings.total_harmonics)]
    )
    return additive_wave_generator

def graph_encoded_and_decoded(encoded, decoded):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(encoded)
    ax1.set_title("Encoded")
    ax1.set_ylabel("Amplitude")

    ax2.plot(decoded)
    ax2.set_title("Decoded")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlabel("Sample index")

    plt.tight_layout()
    plt.show()

def main():
    midi_input: MidiInput = MidiInput()

    payload: Payload = AudioPayload()
    payload.load_from_file(r"assets/sinesweep.wav")

    serializer: Serializer = AudioSerializer(settings, SerializerMode.DIGITAL)

    encoding_strategy: EncodingStrategy = TwoSplitEncodingStrategy(
        settings, sawtooth(settings), serializer
    )
    encoding_strategy.load_payload(payload)

    encoder: Encoder = Encoder(encoding_strategy)

    decoding_strategy: DecodingStrategy = TwoSplitDecodingStrategy(settings, sawtooth(settings))

    framing_sync_controller: FramingSyncController = FramingSyncController()
    sink: Sink = AudioSink(framing_sync_controller, SinkBehaviour.LIVE)

    deserializer: Deserializer = AudioDeserializer(settings, sink, SerializerMode.DIGITAL)
    decoder: Decoder = Decoder(settings, decoding_strategy, deserializer)

    midi_input.on_play(encoder.set_f0)
    midi_input.trigger(2.0)

    num_samples = settings.audio_driver_polling_rate
    encoded = []
    decoded = []
    for i in range(30):
        encoded_frames = encoder.process(num_samples)
        encoded += encoded_frames.get_samples()
        decoded_frames = decoder.process(encoded_frames, num_samples)
        decoded += decoded_frames.get_samples()

    graph_encoded_and_decoded(encoded, decoded)

if __name__ == '__main__':
    main()
