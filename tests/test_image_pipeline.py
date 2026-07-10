"""Serialize -> sink round trip at the symbol-row level (no audio DSP)."""

import pytest
from PIL import Image

from Framing.FramingSyncController import FramingSyncController
from Payload import ImagePayload, SymbolRow
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


def build_pipeline(mode, behaviour, settings, image_path, on_image=None):
    codec = make_pixel_codec(mode, settings)

    payload = ImagePayload(settings, codec)
    payload.load_from_file(image_path)

    serializer = ImageSerializer(settings, mode)
    serializer.load_payload(payload)

    sink = ImageSink(FramingSyncController.from_settings(settings),
                     behaviour, codec, settings, on_image=on_image)
    return payload, serializer, sink


def feed_loops(serializer, sink, settings, loops=1):
    size = serializer._serialized_payload.get_size()
    assert size % settings.data_harmonics == 0
    rows_per_loop = size // settings.data_harmonics
    for _ in range(rows_per_loop * loops):
        sink.push_many([serializer.get_symbol_row(settings.data_harmonics)])


@pytest.mark.parametrize("mode", [SerializerMode.DIGITAL, SerializerMode.ANALOGUE])
def test_live_round_trip_exact(mode, test_image):
    settings = Settings()
    payload, serializer, sink = build_pipeline(
        mode, SinkBehaviour.LIVE, settings, test_image)

    feed_loops(serializer, sink, settings, loops=1)

    latest = sink.get_latest_image()
    assert latest is not None
    pixels, width, height, channels = latest
    assert (width, height, channels) == (settings.image_target_w,
                                         settings.image_target_h,
                                         settings.image_channels)
    assert pixels == payload.get_pixel_bytes()


def test_clean_mode_publishes_once_per_frame(test_image):
    settings = Settings()
    published = []
    payload, serializer, sink = build_pipeline(
        SerializerMode.DIGITAL, SinkBehaviour.CLEAN, settings, test_image,
        on_image=published.append)

    feed_loops(serializer, sink, settings, loops=2)

    assert len(published) == 2  # one publish per finalized frame, not per row
    # First frame adopted directly; identical second frame blends to the same bytes.
    assert published[0][0] == payload.get_pixel_bytes()
    assert published[1][0] == payload.get_pixel_bytes()


def test_live_publishes_progressively(test_image):
    settings = Settings()
    published = []
    payload, serializer, sink = build_pipeline(
        SerializerMode.DIGITAL, SinkBehaviour.LIVE, settings, test_image,
        on_image=published.append)

    feed_loops(serializer, sink, settings, loops=1)

    # Live mode publishes on every data write inside the frame.
    assert len(published) > 100


def test_signal_drop_publishes_merged_partial(test_image):
    settings = Settings()
    published = []
    payload, serializer, sink = build_pipeline(
        SerializerMode.DIGITAL, SinkBehaviour.CLEAN, settings, test_image,
        on_image=published.append)

    size = serializer._serialized_payload.get_size()
    rows_per_loop = size // settings.data_harmonics
    # Feed the start marker plus roughly half of the data rows, then drop.
    for _ in range(rows_per_loop // 2):
        sink.push_many([serializer.get_symbol_row(settings.data_harmonics)])
    sink.on_signal_drop()

    assert len(published) == 1
    pixels = published[0][0]
    expected = payload.get_pixel_bytes()
    # The arrived prefix was adopted directly as the first frame.
    prefix = len(expected) // 3
    assert pixels[:prefix] == expected[:prefix]
