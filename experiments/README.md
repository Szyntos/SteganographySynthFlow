# experiments — measurement harness for the thesis

Headless, reproducible experiments over the real `EncoderDSP`/`DecoderDSP`
pipeline (the exact objects the GUIs drive), so results reflect the full
system: serialization, framing/sync, FIFOs, energy gate, f0 estimation and
drop tolerance.

```
ExperimentConfig ──► run_experiment ──► RunResult ──► metrics / plotting
        │                  │
        │        encode → channel (impairments) → decode
        └──► sweep(base, grid, collect) → rows → save_csv
```

## Quick start

```python
from experiments import ExperimentConfig, run_experiment, channel, metrics

cfg = ExperimentConfig(payload_kind="image", f0=400.0, duration_s=12.0,
                       channel=channel.Chain(channel.AWGN(snr_db=25)))
r = run_experiment(cfg)
print("PSNR:", metrics.image_psnr_db(r.ground_truth_image, r.last_image))
print("BER :", metrics.bit_error_rate(r.ground_truth_image[0], r.last_image and r.last_image[0]))
print("latency to first frame:", r.first_frame_latency_s(), "s")
```

Run examples from the repo root, e.g. `python experiments/examples/ber_vs_snr.py`.

## Modules

| Module        | What it gives you |
|---------------|-------------------|
| `config.py`   | `ExperimentConfig` — one frozen description of a run (payload, strategy, f0, codec, bits/symbol, channel, raw `Settings` overrides). `.with_(...)` for variations. |
| `channel.py`  | Composable impairments: `AWGN`, `Gain`, `HardClip`, `Quantize`, `Butterworth`, `DropBlocks`, `SampleShift`, `ClockSkew`, `Echo`, `Reverb`, combined via `Chain`. All seeded/deterministic. |
| `runner.py`   | `run_experiment(cfg) -> RunResult` (signals, framed outputs with publish timestamps, ground truth, f0/confidence tracks, wall times). `render_carrier(cfg)` for imperceptibility baselines. |
| `metrics.py`  | BER/byte error rate, image MSE/PSNR/SSIM, text CER (Levenshtein), audio RMSE/SNR, embedding SNR + log-spectral distance vs clean carrier, raw bitrate, goodput, f0-tracking error. |
| `sweep.py`    | Cartesian grids over config fields, per-run metric collection, crash-tolerant rows, `save_csv`. |
| `plotting.py` | Metric-vs-parameter curves, carrier/encoded spectrogram pairs, sent/decoded image pairs, f0 tracks. |

## Suggested thesis experiments

**Robustness (payload fidelity vs channel severity)**
- BER / image PSNR / text CER vs AWGN SNR — the waterfall curve, per
  `bits_per_symbol` (1–8) and per strategy (`two` vs `four`).
- Same vs low-pass cutoff (`Butterworth`) — which harmonics are load-bearing;
  relate to `data_offset`/`data_harmonics` and f0 (harmonic 50 at f0=400 Hz
  sits at 20 kHz, right at the band edge).
- Dropout recovery: error vs `DropBlocks(drop_prob)` — validates
  `EnergyGate` + `DropTolerance` (also check `gated_blocks`).
- Alignment sensitivity: error vs `SampleShift(shift)` swept over one
  chunk — how sharp is the sync window the tune slider compensates.
- Clock skew tolerance: error vs `ClockSkew(ppm)` — two-soundcard realism.
- Room effects: `Echo`/`Reverb` severity vs error — the over-the-air story.

**Capacity / rate–distortion trade-off**
- `raw_bitrate_bps` and `goodput_bps` vs `bits_per_symbol` and strategy, with
  the BER at which each operating point still works → the capacity table.
- `phase_range` sweep (via `settings_overrides`): larger phase offsets are
  easier to decode but more audible — plot BER *and* embedding SNR on the
  same x-axis; that trade-off curve is a centrepiece figure.
- DIGITAL vs ANALOGUE codec under identical noise.

**Imperceptibility**
- `embedding_snr_db(render_carrier(cfg), r.encoded)` and
  `log_spectral_distance_db` vs `phase_range`, `bits_per_symbol`,
  `data_harmonics` — how invisible is the embedding.
- Spectrogram pairs (`plot_spectrogram_pair`) for the qualitative figure.

**f0 estimation / musical usability**
- Decoder f0 mismatch: sweep `decoder_f0` around the true `f0`
  (`f0_mode="manual"`) — how many cents of detune the decoder tolerates.
- Estimator comparison: `f0_mode` in {autocorr, fft} under noise —
  `f0_tracking_error_hz` + payload error vs SNR.
- `f0` sweep across the playable range (the 400–480 Hz constraint for the
  image channel): error vs fundamental.

**System behaviour**
- `first_frame_latency_s` (sync acquisition + one frame) vs frame/sync
  settings; `realtime_factor` vs `chunk_size`/strategy for the real-time
  claim.

## Conventions

- One fresh `Settings` per side per run (the runner does this) — never share
  a `Settings` across runs; the facades mutate `chunk_size`.
- Every stochastic impairment takes a `seed`; sweep `repeats=` with
  `AWGN(snr_db=s, seed=rep)` style variation for confidence intervals.
- `binary`/`text` payloads require `payload_path`; `image`/`audio` fall back
  to the default assets.
