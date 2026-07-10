"""Serialize -> sink round trip at the symbol-row level (no audio DSP)."""

import pytest

from Framing.FramingSyncController import FramingSyncController
from Payload import TextPayload
from Payload.pixel_codec import make_pixel_codec
from Serializer import TextSerializer
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import RawTextSink, SinkBehaviour, TextSink

TEST_TEXT = "Hello, world! éèç \U0001F600"


def build_pipeline(mode, behaviour, settings, text, on_text=None):
    codec = make_pixel_codec(mode, settings)

    payload = TextPayload(settings, codec)
    payload.load_from_string(text)

    serializer = TextSerializer(settings, mode)
    serializer.load_payload(payload)

    sink = TextSink(FramingSyncController.from_settings(settings),
                     behaviour, codec, on_text=on_text)
    return payload, serializer, sink


def feed_loops(serializer, sink, settings, loops=1):
    size = serializer._serialized_payload.get_size()
    assert size % settings.data_harmonics == 0
    rows_per_loop = size // settings.data_harmonics
    for _ in range(rows_per_loop * loops):
        sink.push_many([serializer.get_symbol_row(settings.data_harmonics)])


@pytest.mark.parametrize("mode", [SerializerMode.DIGITAL, SerializerMode.ANALOGUE])
def test_round_trip_exact(mode):
    settings = Settings()
    payload, serializer, sink = build_pipeline(
        mode, SinkBehaviour.LIVE, settings, TEST_TEXT)

    feed_loops(serializer, sink, settings, loops=1)

    assert sink.get_text() == TEST_TEXT


def test_clean_sink_publishes_once_per_frame():
    settings = Settings()
    published = []
    payload, serializer, sink = build_pipeline(
        SerializerMode.DIGITAL, SinkBehaviour.LIVE, settings, TEST_TEXT, on_text=published.append)

    feed_loops(serializer, sink, settings, loops=2)

    assert published == [TEST_TEXT, TEST_TEXT]


def test_raw_sink_publishes_progressively_char_by_char():
    settings = Settings()
    codec = make_pixel_codec(SerializerMode.DIGITAL, settings)

    payload = TextPayload(settings, codec)
    payload.load_from_string(TEST_TEXT)

    serializer = TextSerializer(settings, SerializerMode.DIGITAL)
    serializer.load_payload(payload)

    published = []
    raw_sink = RawTextSink(codec, max_chars=200, on_text=published.append)

    size = serializer._serialized_payload.get_size()
    rows_per_loop = size // settings.data_harmonics
    for _ in range(rows_per_loop):
        raw_sink.push_many([serializer.get_symbol_row(settings.data_harmonics)])

    assert len(published) > 1
    assert any(TEST_TEXT in p for p in published)


def test_signal_drop_resets_cleanly():
    settings = Settings()
    payload, serializer, sink = build_pipeline(
        SerializerMode.DIGITAL, SinkBehaviour.LIVE, settings, TEST_TEXT)

    size = serializer._serialized_payload.get_size()
    rows_per_loop = size // settings.data_harmonics
    for _ in range(rows_per_loop // 2):
        sink.push_many([serializer.get_symbol_row(settings.data_harmonics)])
    sink.on_signal_drop()

    assert sink.get_text() is None

    serializer.reset_loop()
    feed_loops(serializer, sink, settings, loops=1)
    assert sink.get_text() == TEST_TEXT
