"""Full end-to-end check: image file -> serialize -> encode to audio ->
decode from audio -> deserialize -> reconstructed image."""

import pytest
from PIL import Image

from AdditiveWaveGenerator import AdditiveWaveGenerator
from Decoder import Decoder, TwoSplitDecodingStrategy
from Deserializer import ImageDeserializer
from Encoder import Encoder, TwoSplitEncodingStrategy
from Framing.FramingSyncController import FramingSyncController
from Payload import ImagePayload
from Payload.pixel_codec import make_pixel_codec
from Serializer import ImageSerializer
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import ImageSink, SinkBehaviour


@pytest.fixture
def test_image(tmp_path):
    path = tmp_path / "test_image.png"
    image = Image.new("RGB", (64, 48))
    pixels = image.load()
    for y in range(48):
        for x in range(64):
            pixels[x, y] = ((x * 4) % 256, (y * 5) % 256, ((x + y) * 3) % 256)
    image.save(path)
    return str(path)


def make_harmonic_generator(settings):
    generator = AdditiveWaveGenerator(settings)
    generator.set_omegas([float(i + 1) for i in range(settings.total_harmonics)])
    generator.set_phases([0.0] * settings.total_harmonics)
    generator.set_amps([settings.base_amplitude / (i + 1) for i in range(settings.total_harmonics)])
    return generator


# f0 constraints: all harmonics below Nyquist (total_harmonics * f0 < fs_out/2,
# so f0 < 480) and at least two fundamental cycles per half-chunk so adjacent
# harmonics sit >= 2 DFT bins apart, outside the Hann window's leakage
# (2 * fs_out/f0 <= chunk_size/2, so f0 >= 400). 400 Hz also divides fs_out
# evenly (120 samples per cycle), keeping every harmonic integer-cycle.
def run_audio_round_trip(mode, settings, image_path, f0=400.0, loops=2):
    codec = make_pixel_codec(mode, settings)

    payload = ImagePayload(settings, codec)
    payload.load_from_file(image_path)

    serializer = ImageSerializer(settings, mode)
    encoding_strategy = TwoSplitEncodingStrategy(
        settings, make_harmonic_generator(settings), serializer)
    encoding_strategy.load_payload(payload)
    encoder = Encoder(encoding_strategy)

    decoding_strategy = TwoSplitDecodingStrategy(settings, make_harmonic_generator(settings))
    # CLEAN publishes once per finalized frame, so get_latest_image() is a
    # complete frame. LIVE publishes after every row write, and the stream may
    # end mid-frame, leaving the latest image a barely-started canvas.
    sink = ImageSink(FramingSyncController.from_settings(settings),
                     SinkBehaviour.CLEAN, codec, settings)
    deserializer = ImageDeserializer(settings, sink, mode)
    decoder = Decoder(settings, decoding_strategy, deserializer)

    encoder.set_f0(f0)
    decoding_strategy.set_f0(f0)

    rows_per_loop = serializer._serialized_payload.get_size() // settings.data_harmonics
    total_samples = rows_per_loop * settings.chunk_size * loops + settings.chunk_size * 4
    block = settings.audio_driver_polling_rate
    for _ in range(total_samples // block + 1):
        encoded = encoder.process(block)
        decoder.process(encoded, block)

    return payload, sink


def byte_accuracy(actual, expected):
    assert len(actual) == len(expected)
    return sum(1 for a, e in zip(actual, expected) if a == e) / len(expected)


def test_digital_audio_round_trip(test_image):
    settings = Settings()
    payload, sink = run_audio_round_trip(SerializerMode.DIGITAL, settings, test_image)

    latest = sink.get_latest_image()
    assert latest is not None, "no image frame was reconstructed from audio"
    pixels, width, height, channels = latest
    assert (width, height, channels) == (settings.image_target_w,
                                         settings.image_target_h,
                                         settings.image_channels)
    accuracy = byte_accuracy(pixels, payload.get_pixel_bytes())
    assert accuracy >= 0.99, f"digital byte accuracy {accuracy:.4f}"


def test_analogue_audio_round_trip(test_image):
    settings = Settings()
    payload, sink = run_audio_round_trip(SerializerMode.ANALOGUE, settings, test_image)

    latest = sink.get_latest_image()
    assert latest is not None, "no image frame was reconstructed from audio"
    pixels, _, _, _ = latest
    expected = payload.get_pixel_bytes()
    mean_abs_error = sum(abs(a - e) for a, e in zip(pixels, expected)) / len(expected)
    assert mean_abs_error < 8.0, f"analogue mean abs error {mean_abs_error:.2f}"
