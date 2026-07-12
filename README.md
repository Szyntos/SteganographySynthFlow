# SteganographySynthFlow

Real-time audio steganography built on an additive synthesizer. A payload
(audio, image, text, or arbitrary binary) is hidden in the **phase offsets of
the harmonics** of a synthesized musical tone: the carrier is a 50-harmonic
additive tone at a playable fundamental `f0` (driven live from a MIDI device
or the computer keyboard), and each chunk of samples modulates the phases of
40 data harmonics to carry one symbol row. A decoder listening to the audio
recovers the phase offsets and reconstructs the payload in real time.

## How it works

- **Carrier** — `AdditiveWaveGenerator` synthesizes `total_harmonics` (50)
  sine partials with 1/n amplitudes at the current `f0`.
- **Embedding** — an `EncodingStrategy` splits each chunk into pilot and data
  regions (`Framing/SplitLayout`). The pilot region carries the clean carrier
  for phase reference; in the data region each data harmonic's phase is
  offset by the symbol value, ramped in/out by an envelope. `TwoSplit` uses
  one pilot/data pair per chunk; `FourSplit` packs two pairs (and therefore
  needs double the chunk size to keep the same DFT bin spacing).
- **Serialization** — `Serializer/` turns a payload into `SymbolRow`s
  (per-harmonic phase offsets), either analogue (audio samples mapped
  directly to phase) or digital (framed bytes at `bits_per_symbol` bits per
  harmonic). `Framing/` adds sync frames so the decoder can lock on.
- **Decoding** — a `DecodingStrategy` estimates the per-harmonic phases via
  DFT against the pilot reference, `FramingSyncController` finds frame
  boundaries, and the recovered symbols flow into a `Sink/` (audio playback,
  image reassembly, text, or raw bytes). `F0Estimator/` tracks the carrier
  fundamental from the received audio; `EnergyGate` and `DropTolerance` keep
  the decoder stable through silence and dropped chunks.
- **Facades** — `EncoderDSP` and `DecoderDSP` assemble the full pipelines
  behind a single API used by the GUIs.

## Running

Requires Python 3.10+ and the packages in `requirements.txt`
(`pip install -r requirements.txt`).

- `python split/encoder_gui.py` — encoder: pick a payload, play notes via
  MIDI/keyboard, outputs the encoded tone to an audio device.
- `python split/decoder_gui.py` — decoder: listens to an input device and
  reconstructs the payload live.
- `python gui.py` — combined encoder+decoder loopback GUI for
  experimentation.
- `python main.py` — headless encode→decode round trip that plots
  encoded/decoded/expected signals.
- `python exp/encode_vs_carrier.py` — dumps diagnostic WAVs (carrier vs.
  encoded vs. recovered) to `exp/output/`.

## Experiments

`exp/harness.py` runs the pipeline offline (no audio device, no GUI) and is
the entry point for parameter sweeps and plots:

```python
from exp.harness import run_round_trip
from Settings import Settings

rt = run_round_trip(settings=Settings(), f0=400.0, strategy_kind="four")
print(rt.rmse())          # decoded vs. expected
rt.encoded, rt.decoded, rt.expected, rt.diff, rt.diff_dc_removed
```

`run_round_trip` mutates the `Settings` it is given (chunk size follows the
strategy), so **pass a fresh `Settings()` per configuration** rather than
sharing one across a sweep.

## Tests

```
python -m pytest
```

## Layout

| Path | Contents |
| --- | --- |
| `Encoder/`, `Decoder/` | Encode/decode strategies (`TwoSplit`, `FourSplit`) |
| `Serializer/`, `Payload/`, `Sink/` | Payload ⇄ symbol-row conversion and output sinks |
| `Framing/` | Pilot/data split layout, frame sync |
| `F0Estimator/` | Autocorrelation & FFT f0 tracking, chromatic quantizer |
| `EncoderDSP.py`, `DecoderDSP.py` | High-level pipeline facades used by the GUIs |
| `Settings.py` | All DSP parameters (sample rate, chunk size, harmonics, bits/symbol) |
| `split/` | Standalone encoder and decoder GUIs |
| `tests/` | Pytest suite (pipelines, framing, codecs) |
