import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from AdditiveWaveGenerator import AdditiveWaveGenerator
from Decoder import Decoder, TwoSplitDecodingStrategy
from Encoder import Encoder, TwoSplitEncodingStrategy
from Framing import FramingSyncController
from Payload import AudioPayload
from Serializer import AudioSerializer
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import SinkBehaviour, AudioSink

_default_settings = Settings()
NUM_ITERS = 80 * 2
NUM_SAMPLES_PER_ITER = _default_settings.audio_driver_polling_rate
TOTAL_SAMPLES = NUM_ITERS * NUM_SAMPLES_PER_ITER


def harmonic_generator(settings: Settings) -> AdditiveWaveGenerator:
    gen = AdditiveWaveGenerator(settings)
    gen.set_omegas([float(i + 1) for i in range(settings.total_harmonics)])
    gen.set_phases([0.0] * settings.total_harmonics)
    gen.set_amps([settings.base_amplitude / (i + 1) for i in range(settings.total_harmonics)])
    return gen


def build_and_run(settings: Settings, f0: float):
    payload = AudioPayload()
    payload.load_from_file(r"assets/sinesweep.wav")

    serializer = AudioSerializer(settings, SerializerMode.DIGITAL)
    encoding_strategy = TwoSplitEncodingStrategy(settings, harmonic_generator(settings), serializer)
    encoding_strategy.load_payload(payload)
    encoder = Encoder(encoding_strategy)
    encoder.set_f0(f0)

    decoding_strategy = TwoSplitDecodingStrategy(settings)
    decoding_strategy.set_f0(f0)
    framing_sync_controller = FramingSyncController()
    sink = AudioSink(framing_sync_controller, SinkBehaviour.LIVE)
    decoder = Decoder(settings, decoding_strategy, sink)

    encoded = []
    decoded = []
    for _ in range(NUM_ITERS):
        enc_chunk = encoder.process(NUM_SAMPLES_PER_ITER)
        encoded.extend(enc_chunk.get_samples())
        dec_chunk = decoder.process(enc_chunk, NUM_SAMPLES_PER_ITER)
        decoded.extend(dec_chunk.get_samples())

    return np.array(encoded), np.array(decoded)


def make_settings_with_chunk_size(chunk_size: int) -> Settings:
    s = Settings()
    s.chunk_size = chunk_size
    s.pilot_size = chunk_size / 2
    s.data_size = chunk_size / 2
    s.MSG_FS = (s.data_harmonics * s.fs_out) / chunk_size
    return s


def precompute_pitch_frames():
    settings = Settings()
    frames = []
    pitches = range(400, 810, 10)
    total = len(list(pitches))
    for idx, f0 in enumerate(pitches):
        print(f"  pitch {f0} Hz  ({idx + 1}/{total})")
        enc, dec = build_and_run(settings, float(f0))
        frames.append((f0, enc, dec))
    return frames


def precompute_chunk_frames():
    frames = []
    chunk_sizes = range(240, 980, 20)
    total = len(list(chunk_sizes))
    for idx, cs in enumerate(chunk_sizes):
        print(f"  chunk_size {cs}  ({idx + 1}/{total})")
        settings = make_settings_with_chunk_size(cs)
        enc, dec = build_and_run(settings, 500.0)
        frames.append((cs, enc, dec))
    return frames


def _sym_lim(arrays, pad=1.1):
    peak = max(np.abs(a).max() for a in arrays)
    return -peak * pad, peak * pad


def save_animations(pitch_frames, chunk_frames):
    x = np.arange(TOTAL_SAMPLES)
    writer = animation.PillowWriter(fps=7)

    # ---- Pitch animation ----
    enc_lim_p = _sym_lim([f[1] for f in pitch_frames])
    dec_lim_p = _sym_lim([f[2] for f in pitch_frames])

    fig1, (ax1e, ax1d) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    fig1.suptitle("Pitch sweep: 400 → 800 Hz (+10 Hz/frame)", fontsize=13)

    line1e, = ax1e.plot(x, pitch_frames[0][1], lw=0.6, color="steelblue")
    ax1e.set_ylim(*enc_lim_p)
    ax1e.set_ylabel("Amplitude")
    title1e = ax1e.set_title(f"Encoded  —  f0 = {pitch_frames[0][0]} Hz")

    line1d, = ax1d.plot(x, pitch_frames[0][2], lw=0.6, color="darkorange")
    ax1d.set_ylim(*dec_lim_p)
    ax1d.set_ylabel("Amplitude")
    ax1d.set_xlabel("Sample index")
    ax1d.set_title("Decoded")

    fig1.tight_layout()

    def update_pitch(i):
        f0, enc, dec = pitch_frames[i]
        line1e.set_ydata(enc)
        line1d.set_ydata(dec)
        title1e.set_text(f"Encoded  —  f0 = {f0} Hz")
        return line1e, line1d, title1e

    anim1 = animation.FuncAnimation(
        fig1, update_pitch, frames=len(pitch_frames), interval=150, blit=True
    )
    print("Saving pitch_animation.gif…")
    anim1.save("pitch_animation.gif", writer=writer)
    plt.close(fig1)

    # ---- Chunk size animation ----
    enc_lim_c = _sym_lim([f[1] for f in chunk_frames])
    dec_lim_c = _sym_lim([f[2] for f in chunk_frames])

    fig2, (ax2e, ax2d) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    fig2.suptitle("Chunk size sweep: 240 → 960 (+20 samples/frame)", fontsize=13)

    line2e, = ax2e.plot(x, chunk_frames[0][1], lw=0.6, color="seagreen")
    ax2e.set_ylim(*enc_lim_c)
    ax2e.set_ylabel("Amplitude")
    title2e = ax2e.set_title(f"Encoded  —  chunk_size = {chunk_frames[0][0]}")

    line2d, = ax2d.plot(x, chunk_frames[0][2], lw=0.6, color="crimson")
    ax2d.set_ylim(*dec_lim_c)
    ax2d.set_ylabel("Amplitude")
    ax2d.set_xlabel("Sample index")
    ax2d.set_title("Decoded")

    fig2.tight_layout()

    def update_chunk(i):
        cs, enc, dec = chunk_frames[i]
        line2e.set_ydata(enc)
        line2d.set_ydata(dec)
        title2e.set_text(f"Encoded  —  chunk_size = {cs}")
        return line2e, line2d, title2e

    anim2 = animation.FuncAnimation(
        fig2, update_chunk, frames=len(chunk_frames), interval=150, blit=True
    )
    print("Saving chunk_size_animation.gif…")
    anim2.save("chunk_size_animation.gif", writer=writer)
    plt.close(fig2)


if __name__ == "__main__":
    print("Precomputing pitch frames (400–800 Hz, step 10)…")
    pitch_frames = precompute_pitch_frames()

    print("Precomputing chunk-size frames (240–960, step 20)…")
    chunk_frames = precompute_chunk_frames()

    save_animations(pitch_frames, chunk_frames)
    print("Done. Saved pitch_animation.gif and chunk_size_animation.gif")
