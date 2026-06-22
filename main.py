from Decoder import Decoder
from Decoder import DecodingStrategy
from Decoder import TwoSplitDecodingStrategy
from Deserializer.Deserializer import Deserializer
from Deserializer.DigitalDeserializer import DigitalDeserializer
from Encoder.Encoder import Encoder
from Encoder.EncodingStrategy import EncodingStrategy
from Encoder.TwoSplitEncodingStrategy import TwoSplitEncodingStrategy
from MidiInput import MidiInput
from AdditiveWaveGenerator import AdditiveWaveGenerator
from Payload import Payload
from Serializer.Serializer import Serializer
from Serializer.DigitalSerializer import DigitalSerializer


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
