"""Synth voice (ADSR envelopes + resonant LPF gain shaping) unit tests plus
an audio round-trip proving envelope/filter motion doesn't corrupt decode."""

import numpy as np
import pytest
from PIL import Image

from AdditiveWaveGenerator import AdditiveWaveGenerator
from Decoder import Decoder, TwoSplitDecodingStrategy
from Encoder import Encoder, TwoSplitEncodingStrategy
from Framing.FramingSyncController import FramingSyncController
from Payload import ImagePayload
from Payload.pixel_codec import make_pixel_codec
from Serializer import ImageSerializer
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import ImageSink, SinkBehaviour
from SynthVoice import ADSREnvelope, SynthVoice, TwoPoleLowPassFilter


# ── ADSR ────────────────────────────────────────────────────────────────────

def advance_n(env, n, dt=0.02, a=0.1, d=0.1, s=0.5, r=0.1):
    value = env.value
    for _ in range(n):
        value = env.advance(dt, a, d, s, r)
    return value


def test_adsr_full_cycle():
    env = ADSREnvelope()
    assert env.state == "idle" and env.value == 0.0

    env.note_on()
    advance_n(env, 5)  # 0.1 s attack at dt=0.02 -> peak
    assert env.value == pytest.approx(1.0)
    advance_n(env, 8)  # decay to sustain (spare steps for float drift)
    assert env.state == "sustain" and env.value == pytest.approx(0.5)

    env.note_off()
    advance_n(env, 20)  # release well past 0.1 s
    assert env.state == "idle" and env.value == 0.0


def test_adsr_retrigger_resets_attack_from_zero():
    env = ADSREnvelope()
    env.note_on()
    advance_n(env, 20)  # settle at sustain
    assert env.state == "sustain"

    env.note_on()  # legato key change: full reset
    value = env.advance(0.02, 0.1, 0.1, 0.5, 0.5)
    assert env.state == "attack"
    assert value == pytest.approx(0.02 / 0.1)  # one attack step up from zero


def test_adsr_zero_times_are_instant():
    env = ADSREnvelope()
    env.note_on()
    env.advance(0.01, 0.0, 0.0, 0.3, 0.0)
    env.advance(0.01, 0.0, 0.0, 0.3, 0.0)
    assert env.value == pytest.approx(0.3)
    env.note_off()
    env.advance(0.01, 0.0, 0.0, 0.3, 0.0)
    assert env.value == 0.0


# ── filter response ─────────────────────────────────────────────────────────

def test_lpf_gains_floor_cap_and_shape():
    filt = TwoPoleLowPassFilter()
    freqs = np.arange(1, 51, dtype=np.float64) * 400.0
    gains = filt.gains(freqs, cutoff_hz=2000.0, q=4.0, gain_cap=4.0, min_gain=0.05)

    assert np.all(gains >= 0.05)          # never zero (decode floor)
    assert np.all(gains <= 4.0)           # resonance peak capped
    assert gains[0] == pytest.approx(1.0, abs=0.1)   # passband ~unity
    assert gains[4] > 1.0                 # resonance peak near cutoff (2 kHz)
    assert gains[-1] == pytest.approx(0.05)  # deep stopband hits the floor


def test_voice_bypass_when_disabled():
    # Disabled voice: amp gain must be exactly 1 and the filter static at the
    # base cutoff. With the cutoff wide open the low harmonics pass at ~unity
    # (the 2-pole response still dips a little approaching the cutoff itself).
    settings = Settings()
    settings.filter_cutoff_hz = settings.fs_out / 2.0
    voice = SynthVoice(settings)
    omegas = [float(i + 1) for i in range(settings.total_harmonics)]
    gains = voice.next_chunk_gains(400.0, omegas, settings.chunk_size)
    assert gains[:10] == pytest.approx(np.ones(10), abs=0.01)
    assert np.all(gains > 0.0)


def test_voice_amp_env_floored_while_held_and_zero_after_release():
    settings = Settings()
    settings.amp_env_sustain = 0.0  # would starve decode without the floor
    settings.amp_env_attack_s = 0.0
    settings.amp_env_decay_s = 0.0
    settings.amp_env_release_s = 0.0
    settings.filter_cutoff_hz = settings.fs_out / 2.0
    voice = SynthVoice(settings)
    voice.set_enabled(True)
    omegas = [1.0]

    voice.note_on()
    voice.next_chunk_gains(400.0, omegas, settings.chunk_size)  # instant attack
    gains = voice.next_chunk_gains(400.0, omegas, settings.chunk_size)
    assert gains[0] == pytest.approx(settings.envelope_min_gain, abs=0.01)

    voice.note_off()
    voice.next_chunk_gains(400.0, omegas, settings.chunk_size)
    gains = voice.next_chunk_gains(400.0, omegas, settings.chunk_size)
    assert gains[0] == 0.0  # release tail may reach true silence


def test_voice_disable_defers_until_release_finishes():
    settings = Settings()
    settings.amp_env_attack_s = 0.0
    settings.amp_env_decay_s = 0.0
    settings.amp_env_sustain = 1.0
    settings.amp_env_release_s = 0.1
    settings.filter_env_release_s = 0.0
    settings.filter_cutoff_hz = settings.fs_out / 2.0
    voice = SynthVoice(settings)
    voice.set_enabled(True)
    voice.note_on()
    omegas = [1.0]
    voice.next_chunk_gains(400.0, omegas, settings.chunk_size)

    # Gate drops (e.g. mouse released on the on-screen piano): must act as a
    # note-off and ring out the release, not snap to bypass gain 1.0.
    voice.set_enabled(False)
    tail = [voice.next_chunk_gains(400.0, omegas, settings.chunk_size)[0]
            for _ in range(4)]
    assert tail[0] < 1.0                      # releasing, not bypassed
    assert all(b <= a for a, b in zip(tail, tail[1:]))  # monotone fade

    for _ in range(20):
        voice.next_chunk_gains(400.0, omegas, settings.chunk_size)
    assert not voice.is_enabled()             # bypass only after the tail


# ── generator amp_gains ─────────────────────────────────────────────────────

def test_amp_gains_of_ones_is_a_noop():
    settings = Settings()
    gen_a = AdditiveWaveGenerator.harmonic(settings)
    gen_b = AdditiveWaveGenerator.harmonic(settings)
    ref = gen_a.generate_block_with_offsets(400.0, 256)
    out = gen_b.generate_block_with_offsets(
        400.0, 256, amp_gains=np.ones(settings.total_harmonics))
    assert out == pytest.approx(ref)


def test_amp_gains_short_array_pads_with_unity():
    settings = Settings()
    gen_a = AdditiveWaveGenerator.harmonic(settings)
    gen_b = AdditiveWaveGenerator.harmonic(settings)
    ref = gen_a.generate_block_with_offsets(400.0, 256)
    out = gen_b.generate_block_with_offsets(400.0, 256, amp_gains=np.ones(3))
    assert out == pytest.approx(ref)


# ── end-to-end: decode survives envelope + swept filter ─────────────────────

def test_round_trip_with_envelope_and_filter_sweep(tmp_path):
    path = tmp_path / "img.png"
    image = Image.new("RGB", (64, 48))
    pixels = image.load()
    for y in range(48):
        for x in range(64):
            pixels[x, y] = ((x * 4) % 256, (y * 5) % 256, ((x + y) * 3) % 256)
    image.save(path)

    settings = Settings()
    # Aggressive but floor-protected voice: low cutoff, hot resonance, and a
    # filter envelope that keeps sweeping the cutoff via repeated retriggers.
    settings.filter_cutoff_hz = 1500.0
    settings.filter_resonance_q = 4.0
    settings.filter_env_amount = 0.6
    settings.filter_env_attack_s = 0.05
    settings.filter_env_decay_s = 0.2
    settings.filter_env_sustain = 0.3
    settings.amp_env_attack_s = 0.02
    settings.amp_env_sustain = 0.7
    settings.validate()

    mode = SerializerMode.DIGITAL
    codec = make_pixel_codec(mode, settings)
    payload = ImagePayload(settings, codec)
    payload.load_from_file(str(path))
    serializer = ImageSerializer(settings, mode)
    encoding_strategy = TwoSplitEncodingStrategy(
        settings, AdditiveWaveGenerator.harmonic(settings), serializer)
    encoding_strategy.load_payload(payload)
    voice = SynthVoice(settings)
    voice.set_enabled(True)
    voice.note_on()
    encoding_strategy.set_synth_voice(voice)
    encoder = Encoder(encoding_strategy)

    decoding_strategy = TwoSplitDecodingStrategy(settings)
    sink = ImageSink(FramingSyncController.from_settings(settings),
                     SinkBehaviour.CLEAN, codec, settings)
    decoder = Decoder(settings, decoding_strategy, sink)

    f0 = 400.0
    encoder.set_f0(f0)
    decoding_strategy.set_f0(f0)

    rows_per_loop = serializer._serialized_payload.get_size() // settings.data_harmonics
    total_samples = rows_per_loop * settings.chunk_size * 2 + settings.chunk_size * 4
    block = settings.audio_driver_polling_rate
    retrigger_every = int(0.5 * settings.fs_out) // block  # sweep keeps moving
    for i in range(total_samples // block + 1):
        if i > 0 and i % retrigger_every == 0:
            voice.note_on()
        encoded = encoder.process(block)
        decoder.process(encoded, block)

    latest = sink.get_latest_image()
    assert latest is not None, "no image frame was reconstructed from audio"
    pixels_out, _, _, _ = latest
    expected = payload.get_pixel_bytes()
    accuracy = sum(1 for a, e in zip(pixels_out, expected) if a == e) / len(expected)
    assert accuracy >= 0.99, f"byte accuracy {accuracy:.4f} under envelope+filter"
