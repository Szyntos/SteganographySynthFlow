from Decoder import Decoder, DecodingStrategy, TwoSplitDecodingStrategy
from Deserializer import Deserializer, ImageDeserializer
from Encoder import Encoder, EncodingStrategy, TwoSplitEncodingStrategy
from Serializer import Serializer
from MidiInput import MidiInput
from AdditiveWaveGenerator import AdditiveWaveGenerator
from Payload import Payload
from Serializer.ImageSerializer import ImageSerializer
from SerializerMode import SerializerMode
from Sink import ImageSink, SinkBehaviour


def main():
    midi_input: MidiInput = MidiInput()
    payload: Payload = Payload()

    additive_wave_generator: AdditiveWaveGenerator = AdditiveWaveGenerator()
    encoding_strategy: EncodingStrategy = TwoSplitEncodingStrategy(additive_wave_generator)
    bits_per_float: int = 2

    serializer: Serializer = ImageSerializer(SerializerMode.DIGITAL, bits_per_float)
    encoder: Encoder = Encoder(serializer, encoding_strategy)

    encoder.set_payload(payload)

    decoding_strategy: DecodingStrategy = TwoSplitDecodingStrategy(additive_wave_generator)
    deserializer: Deserializer = ImageDeserializer(SerializerMode.DIGITAL, bits_per_float)

    sink: ImageSink = ImageSink(SinkBehaviour.LIVE)
    decoder: Decoder = Decoder(deserializer, decoding_strategy, sink)

    midi_input.on_play(encoder.set_f0)

    num_samples = 5

    while True:
        encoded_frames = encoder.process(num_samples)
        decoded_frames = decoder.process(encoded_frames, num_samples)
        print(decoded_frames)

if __name__ == '__main__':
    main()
