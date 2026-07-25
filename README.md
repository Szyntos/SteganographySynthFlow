# SteganographySynthFlow

This is a little synthesizer that hides data inside the sound it makes.

Play a note (mouse, QWERTY keyboard, MIDI controller, whatever) and you get
a normal-sounding additive tone — 50 harmonics stacked on top of a
fundamental. What you don't hear is the 1990's hiphop encoded on their phases.

I built this mostly to see if I could get a covert channel riding on
something that actually sounds like a musical instrument, rather than
noise or a dial-up modem screech. It works.

## Demo (with sound)

[<video src="docs/media/Demo.mp4" controls width="720"></video>](https://github.com/user-attachments/assets/99b069e2-b240-4cd7-a025-2a35f6e783c0
)

## How it actually works

- **The carrier.** `AdditiveWaveGenerator` renders `total_harmonics` (50)
  sine partials at 1/n amplitude on top of whatever `f0` you're currently
  playing. That's the tone you hear - a sawtooth wave.
- **Hiding the data.** Each chunk of audio gets split into a "pilot"
  region (clean carrier, used as a phase reference) and a "data" region
  where 40 of the harmonics get their phase offset by a symbol value,
  ramped in and out so it doesn't click.
- **Turning payloads into symbols.** `Serializer/` converts whatever you're
  hiding into rows of per-harmonic phase offsets — either "analogue" mode
  (audio samples mapped straight to phase) or "digital" mode (bytes framed
  into fixed-size symbols, a few bits per harmonic). `Framing/` adds sync
  markers so the decoder can find its footing without a shared clock.
- **Pulling the data back out.** The decoder estimates each harmonic's
  phase with a DFT against the pilot reference, syncs on the frame markers,
  and hands recovered symbols off to a `Sink/` (playing audio back out,
  rebuilding an image, printing text, whatever). `F0Estimator/` tracks the
  fundamental from the incoming audio so it doesn't need to be told what
  note is playing, and there's an energy gate plus some drop tolerance so
  a bit of silence or a dropped chunk doesn't wreck the stream.
- **Gluing it together.** `EncoderDSP` and `DecoderDSP` wire all of the
  above into something the GUI can just call.

## Running it

You'll need Python 3.10+ and the packages in `requirements.txt`, plus the
project itself installed (editable, so it's importable as `synthflow`):

```
pip install -r requirements.txt
pip install -e .
```

(Everything lives under `src/synthflow/` as a real package — see Layout below.)

- `python -m synthflow.gui` — the main way to use this. Opens an empty rack; drop in
  an Encoder module, a Decoder module, or both. One module alone runs
  against your real audio devices. Both together loop the encoder straight
  into the decoder internally and keep their settings in sync. Whenever an
  encoder is racked, a playable keyboard bar shows up along the bottom
  (mouse clicks, QWERTY, a MIDI device, or a MIDI file all work). Want an
  independent encoder and decoder pair instead? Launch a second instance.

## Poking at it offline

`scripts/exp/harness.py` runs the whole pipeline without touching an audio
device or opening the GUI, which is what I use for parameter sweeps:

```python
from scripts.exp.harness import run_round_trip
from synthflow.core.Settings import Settings

rt = run_round_trip(settings=Settings(), f0=400.0, strategy_kind="four")
print(rt.rmse())          # decoded vs. expected
rt.encoded, rt.decoded, rt.expected, rt.diff, rt.diff_dc_removed
```

For actual thesis measurements (BER vs. SNR, imperceptibility, plots), use
the `experiments/` package instead — `experiments/runner.py` and
`experiments/sweep.py` drive full sweeps and write results/plots to
`results/`; `scripts/exp/harness.py` is the lighter one-off/debugging
harness.

## Tests

```
python -m pytest
```

## Layout

Everything importable lives under `src/synthflow/`:

| Path | What's there |
| --- | --- |
| `core/` | `AdditiveWaveGenerator`, `Settings`, `EncoderDSP`/`DecoderDSP`, `SynthVoice`, `EnergyGate`, `DropTolerance` |
| `Encoder/`, `Decoder/` | The encode/decode strategies (`TwoSplit`, `FourSplit`) |
| `Serializer/`, `Payload/`, `Sink/` | Payload ⇄ symbol-row conversion and where decoded output goes |
| `Framing/` | Pilot/data layout and frame sync |
| `F0Estimator/` | Autocorrelation & FFT pitch tracking, `F0Tracker`, chromatic quantizing |
| `io_devices/` | Keyboard/MIDI note input, audio device I/O |
| `gui/` | The rack GUI — panels stay free of DSP logic, engines stay free of Tk |

Elsewhere in the repo:

| Path | What's there |
| --- | --- |
| `tests/` | Pytest suite covering the pipelines, framing, and codecs |
| `scripts/exp/harness.py` | Lightweight offline encode→decode harness (see above) |
| `experiments/` | Full measurement/sweep infra (BER vs. SNR, imperceptibility, plots) — see `experiments/README.md` |

## A note on the phase trick

Everything that shapes the sound (envelopes, filters, whatever comes next)
has to be a pure per-harmonic *gain* multiplier applied once per chunk —
never a real-time-domain filter, never a change mid-chunk. A real filter
shifts phase, and phase is exactly what's carrying the data, so anything
that touches it mid-stream corrupts the decode. If you're adding a new
effect to the synth voice, that's the one rule not to break.
