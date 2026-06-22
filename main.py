from Decoder import Decoder, DecodingStrategy, TwoSplitDecodingStrategy
from Deserializer import Deserializer, DigitalDeserializer
from Encoder import Encoder, EncodingStrategy, TwoSplitEncodingStrategy
from Serializer import Serializer, DigitalSerializer
from MidiInput import MidiInput
from AdditiveWaveGenerator import AdditiveWaveGenerator
from Payload import Payload


def main():
    midi_input: MidiInput = MidiInput()
    payload: Payload = Payload()

    additive_wave_generator: AdditiveWaveGenerator = AdditiveWaveGenerator()
    encoding_strategy: EncodingStrategy = TwoSplitEncodingStrategy(additive_wave_generator)

    serializer: Serializer = DigitalSerializer()
    encoder: Encoder = Encoder(serializer, encoding_strategy)

    encoder.set_payload(payload)

    decoding_strategy: DecodingStrategy = TwoSplitDecodingStrategy(additive_wave_generator)
    deserializer: Deserializer = DigitalDeserializer()
    decoder: Decoder = Decoder(deserializer, decoding_strategy)

    midi_input.on_play(encoder.set_f0)

    num_samples = 5

    while True:
        encoded_frames = encoder.process(num_samples)
        decoded_frames = decoder.process(encoded_frames, num_samples)
        print(decoded_frames)

if __name__ == '__main__':
    main()
