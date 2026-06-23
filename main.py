from Decoder import Decoder, DecodingStrategy, TwoSplitDecodingStrategy
from Deserializer import Deserializer, ImageDeserializer, AudioDeserializer
from Encoder import Encoder, EncodingStrategy, TwoSplitEncodingStrategy
from Serializer import Serializer, AudioSerializer
from MidiInput import MidiInput
from AdditiveWaveGenerator import AdditiveWaveGenerator
from Payload import Payload, ImagePayload, AudioPayload
from Serializer.ImageSerializer import ImageSerializer
from SerializerMode import SerializerMode
from Sink import ImageSink, SinkBehaviour, AudioSink, Sink


def main():
    midi_input: MidiInput = MidiInput()
    payload: Payload = AudioPayload(512)

    additive_wave_generator_encoding: AdditiveWaveGenerator = AdditiveWaveGenerator()
    additive_wave_generator_decoding: AdditiveWaveGenerator = AdditiveWaveGenerator()
    encoding_strategy: EncodingStrategy = TwoSplitEncodingStrategy(additive_wave_generator_encoding)
    bits_per_symbol: int = 2

    serializer: Serializer = AudioSerializer(SerializerMode.DIGITAL, bits_per_symbol)
    encoder: Encoder = Encoder(serializer, encoding_strategy)

    encoder.set_payload(payload)

    decoding_strategy: DecodingStrategy = TwoSplitDecodingStrategy(additive_wave_generator_decoding)
    deserializer: Deserializer = AudioDeserializer(SerializerMode.DIGITAL, bits_per_symbol)

    sink: Sink = AudioSink(SinkBehaviour.LIVE)
    decoder: Decoder = Decoder(deserializer, decoding_strategy, sink)

    midi_input.on_play(encoder.set_f0)

    num_samples = 5

    while True:
        encoded_frames = encoder.process(num_samples)
        decoded_frames = decoder.process(encoded_frames, num_samples)
        print(decoded_frames)

if __name__ == '__main__':
    main()
