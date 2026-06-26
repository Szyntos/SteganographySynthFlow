from Codec.Audio import AudioDecoderCodec, AudioEncoderCodec
from Decoder import Decoder, DecodingStrategy, TwoSplitDecodingStrategy
from Encoder import Encoder, EncodingStrategy, TwoSplitEncodingStrategy
from Framing import FramingSyncController
from MidiInput import MidiInput
from AdditiveWaveGenerator import AdditiveWaveGenerator
from Payload import AudioPayload
from Serializer import Serializer, AudioSerializer
from SerializerMode import SerializerMode
from Sink import SinkBehaviour, AudioSink


def main():
    midi_input: MidiInput = MidiInput()

    additive_wave_generator_encoding: AdditiveWaveGenerator = AdditiveWaveGenerator()

    bits_per_symbol: int = 2

    encoder_codec = AudioEncoderCodec(SerializerMode.DIGITAL, bits_per_symbol)
    encoding_strategy: EncodingStrategy = TwoSplitEncodingStrategy(additive_wave_generator_encoding)

    encoder: Encoder = Encoder(encoder_codec, encoding_strategy)
    encoder.set_payload(AudioPayload())


    additive_wave_generator_decoding: AdditiveWaveGenerator = AdditiveWaveGenerator()

    decoding_strategy: DecodingStrategy = TwoSplitDecodingStrategy(additive_wave_generator_decoding)

    framing_sync_controller: FramingSyncController = FramingSyncController()
    decoder_codec = AudioDecoderCodec(SerializerMode.DIGITAL, bits_per_symbol, SinkBehaviour.LIVE, framing_sync_controller)
    decoder: Decoder = Decoder(decoder_codec, decoding_strategy)

    midi_input.on_play(encoder.set_f0)
    midi_input.trigger(440.0)

    num_samples = 5

    while True:
        encoded_frames = encoder.process(num_samples)
        decoded_frames = decoder.process(encoded_frames, num_samples)
        print(decoded_frames)

if __name__ == '__main__':
    main()
