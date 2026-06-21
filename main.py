from AudioDevice import AudioDevice
from Encoder import Encoder
from MidiInput import MidiInput
from AdditiveWaveGenerator import AdditiveWaveGenerator
from Payload import Payload


def main():
    midi_input: MidiInput = MidiInput()
    payload: Payload = Payload()
    additive_wave_generator: AdditiveWaveGenerator = AdditiveWaveGenerator()
    encoder: Encoder = Encoder(additive_wave_generator)
    encoder.set_payload(payload)
    midi_input.on_play(encoder.set_f0)

    audio_device: AudioDevice = AudioDevice()
    while True:
        audio_device.audioCallback(encoder, 12)

if __name__ == '__main__':
    main()
