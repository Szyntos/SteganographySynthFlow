# TODO: Volume ADSR + Low-Pass Filter (with Resonance + Filter ADSR)

> **Status: implemented** (SynthVoice.py, tests/test_synth_voice.py).
> Deviations from the plan below, by design decision: cutoff is absolute Hz
> (10 Hz … Nyquist) with no key tracking; env amount is bipolar with
> log-space cutoff modulation; envelopes bypass entirely when no note
> source is active.

Goal: give the encoder synth an amp envelope and a filter envelope/resonance
without corrupting decode. Synth is **monophonic** (one voice, one f0 at a
time) — no voice allocation/stealing to worry about, just one ADSR state
machine per envelope.

## 0. Why this is safe *if done right* (read before implementing)

Decoding (`Decoder/DecodingStrategy.py:_decode`) is **phase-differential per
harmonic**: it compares the phase of harmonic `h` between the `pilot`
segment and the `data` segment of the same chunk
(`Framing/SplitLayout.py` — `pilot_start`/`data_start` are time offsets
within one `chunk_size`, not different harmonics). Symbol value comes from
`angle(Z_d * conj(Z_p))`, so uniform amplitude scaling of a harmonic cancels
out of the math entirely — pure gain changes do not corrupt phase.

Both effects below (ADSR, filter+resonance) are implemented as **per-harmonic
gain multipliers in the additive synth**, computed and applied to
`AdditiveWaveGenerator`'s harmonic amplitudes before summing. They must
never be implemented as real-time-domain filters (biquad/IIR) on the
rendered waveform — a real filter imparts frequency-dependent *phase* shift
(worst right at resonance), which directly corrupts the phase-difference
measurement. Gain-only, always.

Two hazards to guard against for both envelopes:

1. **Non-stationarity within one analysis window.** `_decode` performs a
   windowed DFT separately over the pilot segment and the data segment of a
   single chunk (see `_get_analysis_arrays`, `window_size = internal_clock //
   layout.phases`). If gain changes *during* one of those segments, the
   harmonic's energy leaks into the DFT bins of neighboring harmonics,
   biasing their phase too. **Update gains once per chunk (`chunk_size`
   samples), held constant across the whole chunk** — never ramp
   sample-by-sample. This matches the existing per-chunk granularity of
   `EncodingStrategy.generate_samples`.
2. **Starving data harmonics below the decode floor.** `_mag_threshold =
   1e-6` in the decoder: any harmonic whose magnitude drops under it is
   marked invalid for that chunk and the decoder just holds its previous
   value (`data_valid` mask in `Decoder/DecodingStrategy.py:202`) — not fatal, but it
   stalls that harmonic's data. Worse, `EnergyGate` in `DecoderDSP.py`
   gates the *whole chunk* if overall RMS falls below its floor relative to
   an EMA — a deep envelope/filter dip across many harmonics at once can
   look like silence and drop the chunk. **Both envelopes must be clamped
   so they never fully zero out `data_offset:data_offset+data_harmonics`
   (currently harmonics `10..49` of `50` — `data_offset=10`,
   `data_harmonics=40`, see `DspSettings`), and total energy should stay
   above `EnergyGate`'s floor.**
3. **Starving the bias-reference harmonics.** The decoder also uses the
   *non-data* harmonics (indices `0..data_offset-1`, i.e. `0..9`) as a
   phase-bias reference (`bias_mask` in `DecodingStrategy._decode`, ~line
   188): any of them above `_mag_threshold` contribute to a bias estimate
   that corrects all data phases. A low-pass filter naturally preserves
   these (they're the *lowest* harmonics), but the envelope floor from
   hazard #2 must apply to them too — never carve out "only data harmonics
   get the floor."

Everything below is designed around those constraints.

## 1. Volume (amplitude) ADSR

### 1.1 Where it plugs in

`AdditiveWaveGenerator.generate_block_with_offsets` already accepts
`amp_offsets`, but it's **additive** (`effective_amps = amps +
amp_offsets`), which isn't the right shape for envelope gain. Add a
multiplicative path instead:

- New param `amp_gains: Optional[np.ndarray]` (length == `total_harmonics`,
  broadcastable) alongside `amp_offsets`.
- `effective_amps = (amps + amp_offsets_or_zero) * (amp_gains if given else 1.0)`
- If a shorter `amp_gains` is passed, **pad with `1.0`**, not `0.0` — the
  existing `amp_offsets`/`phase_offsets` padding uses zeros because they're
  additive; a zero pad on a multiplicative array silently mutes harmonics.
- Keep it a per-call array like `phase_offsets`, not a stateful attribute —
  `EncodingStrategy` computes it fresh once per chunk and passes it in,
  same pattern as `phase_offsets`/`phase_envelope`.
- Apply it *before* the existing Nyquist `valid_mask` mute (order doesn't
  change the result, but keep the mask as the final word).

### 1.2 ADSR state machine

New class, e.g. `Envelope/ADSR.py`:

```
class ADSREnvelope:
    def __init__(self, attack_s, decay_s, sustain_level, release_s, fs, update_period_samples):
        ...
    def note_on(self): ...
    def note_off(self): ...
    def value(self) -> float:  # current gain in [0, 1], advances internal clock by update_period_samples
```

- Since the synth is monophonic, this is a single instance with a single
  `note_on`/`note_off` — no polyphonic voice management needed. Put it in a
  tk-free module (envelope logic is DSP, and the gui package split is
  engines-without-tk vs panels-without-DSP).
- **Trigger points — use the real note gate.** The project already has note
  sources (`PianoKeyboard`/`KeyboardNoteInput`/`MidiDeviceInput` →
  `NoteState` → `EncoderDSP.set_f0`). Fire `note_on()` when a key/MIDI note
  goes down and `note_off()` when the last held note is released — that's
  what makes this feel like a synth instead of a fader. Retrigger
  (`note_on` while in decay/sustain/release) should restart the attack
  **from the envelope's current value**, not from zero, so legato playing
  doesn't dip. Fallback when no note source is active (pure payload
  streaming): `note_on()` on first `process()` call, sustain indefinitely,
  `note_off()` only on explicit stop — Attack/Release become the stream's
  fade-in/out.
- **Advance and latch at the chunk boundary** — exactly the pattern
  `set_f0`/`_pending_f0` already uses in `EncodingStrategy`. Note that
  `generate_samples` iterates in *segments* (`segment_len = min(remaining,
  internal_clock - clock_position)`), which can be smaller than a chunk, so
  "once per chunk" concretely means: sample `.value()` (and apply pending
  `note_on`/`note_off` edges) only inside the existing
  `if self._clock_position == 0:` block, and reuse that latched gain for
  every segment until the clock wraps again. Never advance per
  `generate_samples` call — call sizes follow the audio driver, not the
  chunk grid. If a smoother sub-chunk envelope is wanted later, it must
  still be constant *within* each pilot/data segment
  (`chunk_size // layout.phases`) — start with per-chunk, it's the safe
  default. (At `chunk_size=960`, per-chunk updating is a ~50 Hz control
  rate — normal synth modulation territory, no audible zipper.)

### 1.3 Applying the gain

- Sustain level and attack/decay floor must respect hazard #2: pick
  `sustain_level` and clamp `value()` to a configurable **minimum floor**
  (e.g. `settings.envelope_min_gain = 0.15`) so the fully-attacked-but-quiet
  portions of the envelope never approach `EnergyGate`'s `abs_floor`/
  `drop_ratio`. During Attack/Release ramps that *do* need to pass near
  zero (e.g. fade to true silence at stream end), that's fine — it's
  expected to gate as silence there, same as natural signal start/stop.
- Apply the same scalar gain to **all harmonics uniformly**
  (`amp_gains = np.full(total_harmonics, envelope.value())`) — this is the
  "global envelope" case already reasoned about; safe by construction since
  it's uniform, chunk-stationary, and floor-clamped.

### 1.4 Settings additions (`Settings.py`)

`Settings` is a composition root of sub-config groups, and its
`__setattr__` **refuses unknown flat names** — you cannot just bolt
attributes on. Add a new group class and register it in `_groups()` and
`_OWN_ATTRS` (flat delegation then makes `settings.amp_env_attack_s` work
everywhere for free):

```
class SynthVoiceSettings:
    def __init__(self):
        self.amp_env_attack_s: float = 0.05
        self.amp_env_decay_s: float = 0.1
        self.amp_env_sustain: float = 0.8
        self.amp_env_release_s: float = 0.2
        self.envelope_min_gain: float = 0.15   # floor shared by amp + filter envelopes
        # (§2.5's filter fields live here too)
    def validate(self) -> None: ...
```

### 1.5 Tests

- Round-trip test (encode with envelope on → decode) asserting zero/low bit
  error once past the attack ramp, using the project's existing
  encode/decode harness.
- Assert `sustain * base_amplitude` and `envelope_min_gain` keep harmonic
  magnitude above `_mag_threshold` and chunk RMS above `EnergyGate.abs_floor`
  at the configured `bits_per_symbol`/`data_harmonics`.

## 2. Low-pass filter with resonance (per-harmonic gain shaping)

### 2.1 Model

Since harmonics are discrete (`omega_k = (k+1) * f0`, `k=0..total_harmonics-1`,
see `AdditiveWaveGenerator.harmonic`), "filtering" is just evaluating a
filter's magnitude response `|H(f)|` at each harmonic's frequency and using
that as `amp_gains[k]`. No convolution, no phase — see §0.

Use the standard 2-pole resonant LPF magnitude response (matches the
sound of an analog-modeled SVF/ladder filter without implementing one):

```
f_k = (k + 1) * f0          # harmonic frequency in Hz
x = f_k / cutoff_hz
# 2-pole resonant LPF magnitude, Q controls resonance peak height/sharpness
gain_k = 1.0 / sqrt((1 - x**2)**2 + (x / Q)**2)
gain_k = min(gain_k, resonance_gain_cap)   # clamp the peak near x≈1
```

Normalize so `gain_k <= 1.0` well below cutoff (divide by the DC value,
which is `1.0` here) and cap the resonance peak (`resonance_gain_cap`, e.g.
4.0) so a high-Q peak near `data_offset`'s harmonics can't blow past
`base_amplitude` limits or clip.

**Key tracking (recommended, and very synth):** express cutoff as a
multiple of f0 instead of absolute Hz — `cutoff_hz = cutoff_ratio * f0`.
Then `x = (k + 1) * f0 / (cutoff_ratio * f0) = (k + 1) / cutoff_ratio`,
i.e. the whole gain curve is **f0-independent**: the filter shape follows
the note (100% key tracking, like a classic mono synth), the timbre stays
identical across pitches, and the min-cutoff/decode-floor validation in
§2.2 only has to be checked once instead of across the entire f0 range.
Bonus: `amp_gains` only needs recomputing when the envelope moves, not on
every pitch change. Offer absolute-Hz mode as an option if wanted, but make
ratio mode the default.

### 2.2 Interaction with the data band — the real risk here

`data_offset=10`, `data_harmonics=40`, `total_harmonics=50` (current
`DspSettings`): harmonics 10..49 — the upper 80% of the series — carry
data, and the bottom 10 are the decoder's bias reference (hazard #3). That
means a swept low cutoff will attenuate high data-harmonic indices toward
`_mag_threshold`, stalling their decode (hazard #2), and if `cutoff_hz`
sweeps low enough, most of the data band goes quiet at once and
`EnergyGate` may gate the chunk entirely.

Mitigations, pick per taste but implement at least the floor:

- **Cutoff floor**: clamp `cutoff_hz >= min_cutoff_hz` such that at
  `min_cutoff_hz` every data harmonic's `gain_k * base_amplitude/(k+1)`
  stays above `_mag_threshold` (compute this at settings-validation time
  from `total_harmonics`, `f0` range, and `base_amplitude`, similar to how
  `Settings.validate()` already cross-checks `data_offset + data_harmonics
  <= total_harmonics`).
- **Per-harmonic gain floor**: `gain_k = max(gain_k, per_harmonic_min_gain)`
  so no single data harmonic ever fully dies even at extreme cutoff —
  trades filter "purity" for decode robustness. Recommended default,
  since it's cheap insurance and barely audible.
- Note existing `AdditiveWaveGenerator.generate_block_with_offsets` already
  zeroes harmonics above Nyquist (`valid_mask = omegas*f0 <= nyquist`) —
  same pattern, just add the filter's own floor on top.

### 2.3 Applying it

- Compute `amp_gains[k] = filter_gain_k * envelope_gain` (multiply the ADSR
  scalar from §1 with the per-harmonic filter curve from §2.1) once per
  chunk, pass as the single `amp_gains` array into
  `generate_block_with_offsets` (§1.1's new param) — one multiplicative
  array covers both effects, no separate application passes needed.
- Same **chunk-granularity update** rule as §1.2 (hazard #1): recompute
  `cutoff_hz` (and thus every `gain_k`) once per chunk, hold for the whole
  chunk's `generate_block_with_offsets` call.

### 2.4 Filter envelope (cutoff ADSR)

A second, independent `ADSREnvelope` instance (§1.2's class, reused)
modulates `cutoff_hz` instead of amplitude:

```
cutoff_hz = base_cutoff_hz + filter_env.value() * filter_env_amount_hz
cutoff_hz = clip(cutoff_hz, min_cutoff_hz, nyquist_guard_hz)
```

(In key-tracked ratio mode from §2.1, modulate the *ratio* instead:
`cutoff_ratio = base_cutoff_ratio + filter_env.value() * env_amount_ratio`,
same clip idea in ratio units — this keeps the f0-independence.)

- `filter_env.note_on()`/`note_off()` fire on the same trigger points as
  the amp envelope (§1.2's note gate) — same events, but keep them as
  separate instances so attack/decay/sustain/release timing can differ
  (classic synth voice: amp envelope and filter envelope have independent
  shapes). Same chunk-boundary latch: sample it in the
  `_clock_position == 0` block only.
- `nyquist_guard_hz`: keep comfortably under `fs_out/2` (`Settings.fs_out =
  48000` → guard around, say, `20000`) so the filter's own sweep doesn't
  interact with `AdditiveWaveGenerator`'s existing Nyquist muting in a way
  that causes harmonics to blink in/out chunk-to-chunk (that blinking
  would itself look like hazard #1/#2 to the decoder — treat cutoff
  envelope range as "sweeps within the data band," not up into aliasing
  territory).

### 2.5 Settings additions (`Settings.py`)

Same `SynthVoiceSettings` group as §1.4 (register once, both features'
fields live together):

```
self.filter_key_tracking: bool = True       # ratio mode (§2.1) vs absolute Hz
self.filter_base_cutoff_ratio: float = 20.0 # cutoff = ratio * f0 when key-tracked
self.filter_env_amount_ratio: float = 10.0
self.filter_base_cutoff_hz: float = 8000.0  # used when key_tracking is False
self.filter_env_amount_hz: float = 4000.0
self.filter_resonance_q: float = 1.5
self.filter_resonance_gain_cap: float = 4.0
self.filter_min_cutoff_hz: float = 1500.0   # derive/validate against mag_threshold headroom
self.filter_per_harmonic_min_gain: float = 0.05
self.filter_env_attack_s: float = 0.02
self.filter_env_decay_s: float = 0.15
self.filter_env_sustain: float = 0.4
self.filter_env_release_s: float = 0.2
```

Add a `validate()` check (next to the existing ones in `Settings.py`) that
`filter_min_cutoff_hz` and `filter_per_harmonic_min_gain` together keep the
worst-case data harmonic above `_mag_threshold` — fail loud at config time
rather than silently degrading decode later.

### 2.6 Tests

- Sweep `cutoff_hz` across the full data-harmonic range in a round-trip
  test; assert decode stays correct throughout (not just at rest).
  Round-trip the payload while simultaneously sweeping and confirm the
  reconstructed payload is bit-exact (or within the existing tolerance
  used by other strategy tests).
- Regression test that a high-Q resonance peak parked directly on a data
  harmonic doesn't clip (`gain_k * base_amplitude/(k+1)` summed across all
  harmonics stays within the sample range the rest of the pipeline
  assumes).

## 3. Suggested implementation order

1. Add `amp_gains` multiplicative param to `AdditiveWaveGenerator` (§1.1) —
   small, isolated, testable on its own (unit test: gain array of all 1.0s
   is a no-op vs. current behavior).
2. Add `ADSREnvelope` class + settings (§1.2, §1.4), wire into
   `EncodingStrategy.generate_samples` as a uniform `amp_gains` array.
   Round-trip test.
3. Add filter magnitude-response function + settings (§2.1, §2.5), wire in
   as a second per-harmonic array multiplied into the same `amp_gains`.
   Round-trip test with static cutoff.
4. Add filter envelope (§2.4) modulating cutoff per chunk. Round-trip test
   with a swept cutoff.
5. Hook envelope `note_on`/`note_off` to the existing note sources
   (keyboard/MIDI → `EncoderDSP`), with retrigger-from-current-value.
6. Wire GUI controls (attack/decay/sustain/release ×2, cutoff/ratio,
   resonance, key-track toggle) as a new panel in the `gui` rack package
   (panels stay DSP-free; envelope/filter math lives with the engines),
   exposed through `EncoderDSP` real-time setters following the existing
   `set_f0`/`set_bits_per_symbol` pattern. Parameter changes from the GUI
   should latch at the chunk boundary like `_pending_f0` — a mid-chunk
   cutoff jump is hazard #1 by another name.
