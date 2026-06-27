from Decoder import Decoder, DecodingStrategy, TwoSplitDecodingStrategy
from Deserializer import Deserializer, AudioDeserializer
from Encoder import Encoder, EncodingStrategy, TwoSplitEncodingStrategy
from Framing import FramingSyncController
from MidiInput import MidiInput
from AdditiveWaveGenerator import AdditiveWaveGenerator
from Payload import AudioPayload
from Serializer import AudioSerializer, Serializer
from SerializerMode import SerializerMode
from Sink import SinkBehaviour, AudioSink, Sink


def main():
    midi_input: MidiInput = MidiInput()

    additive_wave_generator_encoding: AdditiveWaveGenerator = AdditiveWaveGenerator()

    bits_per_symbol: int = 2

    num_rows = 10

    serializer: Serializer = AudioSerializer(SerializerMode.DIGITAL, bits_per_symbol)

    encoding_strategy: EncodingStrategy = TwoSplitEncodingStrategy(
        additive_wave_generator_encoding, serializer, num_rows
    )
    encoding_strategy.load_payload(AudioPayload())

    encoder: Encoder = Encoder(encoding_strategy)


    max_driver_block_size = 4096

    additive_wave_generator_decoding: AdditiveWaveGenerator = AdditiveWaveGenerator()

    decoding_strategy: DecodingStrategy = TwoSplitDecodingStrategy(additive_wave_generator_decoding, num_rows)

    framing_sync_controller: FramingSyncController = FramingSyncController()
    sink: Sink = AudioSink(framing_sync_controller, SinkBehaviour.LIVE)

    deserializer: Deserializer = AudioDeserializer(sink, SerializerMode.DIGITAL, bits_per_symbol)
    decoder: Decoder = Decoder(decoding_strategy, deserializer, sink, max_driver_block_size)

    midi_input.on_play(encoder.set_f0)
    midi_input.trigger(440.0)

    num_samples = 20

    while True:
        encoded_frames = encoder.process(num_samples)
        decoded_frames = decoder.process(encoded_frames, num_samples)
        print(decoded_frames)

if __name__ == '__main__':
    main()
