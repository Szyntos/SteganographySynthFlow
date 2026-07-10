"""Serialize -> deserialize round trip at the symbol-row level (no audio DSP)."""

import pytest

from Deserializer import BinaryDeserializer
from Framing.FramingSyncController import FramingSyncController
from Payload import BinaryPayload
from Payload.pixel_codec import make_pixel_codec
from Serializer import BinarySerializer
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import BinarySink, RawBinarySink, SinkBehaviour


@pytest.fixture
def test_binary_file(tmp_path):
    path = tmp_path / "test_binary.bin"
    path.write_bytes(bytes(range(256)) * 3)
    return str(path)


def build_pipeline(mode, behaviour, settings, file_path, on_data=None):
    codec = make_pixel_codec(mode, settings)

    payload = BinaryPayload(settings, codec)
    payload.load_from_file(file_path)

    serializer = BinarySerializer(settings, mode)
    serializer.load_payload(payload)

    sink = BinarySink(FramingSyncController.from_settings(settings),
                       behaviour, codec, on_data=on_data)
    deserializer = BinaryDeserializer(settings, sink, mode)
    return payload, serializer, sink, deserializer


def feed_loops(serializer, deserializer, settings, loops=1):
    size = serializer._serialized_payload.get_size()
    assert size % settings.data_harmonics == 0
    rows_per_loop = size // settings.data_harmonics
    for _ in range(rows_per_loop * loops):
        deserializer.deserialize_symbols([serializer.get_symbol_row(settings.data_harmonics)])


@pytest.mark.parametrize("mode", [SerializerMode.DIGITAL, SerializerMode.ANALOGUE])
def test_round_trip_exact(mode, test_binary_file):
    settings = Settings()
    payload, serializer, sink, deserializer = build_pipeline(
        mode, SinkBehaviour.LIVE, settings, test_binary_file)

    feed_loops(serializer, deserializer, settings, loops=1)

    assert sink.get_bytes() == payload.get_raw_bytes()


def test_raw_sink_publishes_progressively(test_binary_file):
    settings = Settings()
    codec = make_pixel_codec(SerializerMode.DIGITAL, settings)

    payload = BinaryPayload(settings, codec)
    payload.load_from_file(test_binary_file)

    serializer = BinarySerializer(settings, SerializerMode.DIGITAL)
    serializer.load_payload(payload)

    published = []
    raw_sink = RawBinarySink(codec, max_bytes=1024, on_data=published.append)

    size = serializer._serialized_payload.get_size()
    rows_per_loop = size // settings.data_harmonics
    for _ in range(rows_per_loop):
        raw_sink.push_many([serializer.get_symbol_row(settings.data_harmonics)])

    assert len(published) > 1
    assert any(payload.get_raw_bytes() in p for p in published)


def test_signal_drop_resets_cleanly(test_binary_file):
    settings = Settings()
    payload, serializer, sink, deserializer = build_pipeline(
        SerializerMode.DIGITAL, SinkBehaviour.LIVE, settings, test_binary_file)

    size = serializer._serialized_payload.get_size()
    rows_per_loop = size // settings.data_harmonics
    for _ in range(rows_per_loop // 2):
        deserializer.deserialize_symbols([serializer.get_symbol_row(settings.data_harmonics)])
    sink.on_signal_drop()

    assert sink.get_bytes() is None

    serializer.reset_loop()
    feed_loops(serializer, deserializer, settings, loops=1)
    assert sink.get_bytes() == payload.get_raw_bytes()
