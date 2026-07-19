"""Monophonic synth voice: amp ADSR + filter ADSR driving a per-harmonic
gain curve applied inside AdditiveWaveGenerator.

Everything here is gain-only, evaluated once per chunk and held constant for
the whole chunk. It must never become a time-domain filter or a per-sample
ramp: the decoder is phase-differential per harmonic and windows the pilot
and data halves of each chunk separately, so any phase shift or intra-chunk
gain motion corrupts decode (see TODO_filter_volume_envelope.md §0).

No tkinter in this module (engine-side DSP).
"""

import math
from typing import Optional, Sequence

import numpy as np

from Settings import Settings

_ACTIVE_STATES = ("attack", "decay", "sustain")
_EPS = 1e-6


class ADSREnvelope:
    """Chunk-rate linear ADSR. note_on/note_off only set pending edges;
    edges are applied and time advances only inside advance(), which the
    encoding strategy calls exactly once per chunk boundary — so the value
    is guaranteed stationary within a chunk."""

    def __init__(self):
        self._state: str = "idle"
        self._value: float = 0.0
        self._release_start: float = 0.0
        self._pending_on: bool = False
        self._pending_off: bool = False

    @property
    def state(self) -> str:
        return self._state

    @property
    def value(self) -> float:
        return self._value

    def note_on(self) -> None:
        self._pending_on = True
        self._pending_off = False

    def note_off(self) -> None:
        self._pending_off = True

    def reset(self) -> None:
        self._state = "idle"
        self._value = 0.0
        self._pending_on = False
        self._pending_off = False

    def advance(self, dt: float, attack_s: float, decay_s: float,
                sustain: float, release_s: float) -> float:
        """Apply pending edges, step dt seconds, return the new value in [0, 1]."""
        if self._pending_on:
            # Every note-on (including a legato key change) restarts the
            # attack from zero — a full envelope reset per key.
            self._state = "attack"
            self._value = 0.0
            self._pending_on = False
        if self._pending_off:
            if self._state in _ACTIVE_STATES:
                self._state = "release"
                self._release_start = self._value
            self._pending_off = False

        if self._state == "attack":
            if attack_s <= _EPS:
                self._value = 1.0
            else:
                self._value += dt / attack_s
            if self._value >= 1.0:
                self._value = 1.0
                self._state = "decay"
        elif self._state == "decay":
            if decay_s <= _EPS:
                self._value = sustain
            else:
                self._value -= dt * (1.0 - sustain) / decay_s
            if self._value <= sustain:
                self._value = sustain
                self._state = "sustain"
        elif self._state == "sustain":
            self._value = sustain
        elif self._state == "release":
            if release_s <= _EPS:
                self._value = 0.0
            else:
                self._value -= dt * max(self._release_start, _EPS) / release_s
            if self._value <= 0.0:
                self._value = 0.0
                self._state = "idle"
        return self._value


class TwoPoleLowPassFilter:
    """Classic 2-pole resonant LPF *magnitude response* sampled at the
    harmonic frequencies. Gain-only by construction — no phase, no state."""

    kind = "lpf"

    def gains(self, freqs_hz: np.ndarray, cutoff_hz: float, q: float,
              gain_cap: float, min_gain: float) -> np.ndarray:
        x = np.asarray(freqs_hz, dtype=np.float64) / max(cutoff_hz, _EPS)
        response = 1.0 / np.sqrt((1.0 - x * x) ** 2 + (x / max(q, _EPS)) ** 2)
        response = np.minimum(response, gain_cap)
        # Decode floor: a data harmonic under the decoder's magnitude
        # threshold stalls, so the filter may attenuate but never mute.
        return np.maximum(response, min_gain)


FILTER_KINDS = {TwoPoleLowPassFilter.kind: TwoPoleLowPassFilter}


class SynthVoice:
    """Owns both envelopes and the filter; produces the per-harmonic
    amp_gains array once per chunk. Reads all parameters live from the
    settings group so GUI slider writes need no plumbing."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._amp_env = ADSREnvelope()
        self._filter_env = ADSREnvelope()
        self._filter = TwoPoleLowPassFilter()
        self._enabled: bool = False
        self._disable_pending: bool = False
        self._last_filter_gains: Optional[np.ndarray] = None
        self._last_filter_env_value: float = 0.0

    # ── control (called from the audio callback / engine thread) ────────────
    def set_enabled(self, enabled: bool) -> None:
        """Gate sources (mouse piano, MIDI toggle) can drop the note gate the
        instant a key is released — bypassing right then would cut the
        release tails and snap volume/cutoff back to the raw carrier. So a
        disable is deferred: it acts as a note-off, and the voice only truly
        bypasses once the amp envelope has finished releasing."""
        if enabled:
            self._disable_pending = False
            self._enabled = True
        elif self._enabled and not self._disable_pending:
            self._amp_env.note_off()
            self._filter_env.note_off()
            self._disable_pending = True

    def is_enabled(self) -> bool:
        return self._enabled

    def note_on(self) -> None:
        self._amp_env.note_on()
        self._filter_env.note_on()

    def note_off(self) -> None:
        self._amp_env.note_off()
        self._filter_env.note_off()

    def set_filter(self, filt) -> None:
        """Swap the filter model; anything with a matching gains() works."""
        self._filter = filt

    def set_filter_kind(self, kind: str) -> None:
        self._filter = FILTER_KINDS[kind]()

    # ── cutoff law ──────────────────────────────────────────────────────────
    def _modulated_cutoff_hz(self, env_value: float) -> float:
        s = self._settings
        lo = s.filter_min_cutoff_hz
        hi = s.fs_out / 2.0
        # Bipolar, log-space modulation: amount = ±1 can sweep the cutoff
        # across the whole audible range from wherever the base sits.
        log_cutoff = (math.log(max(s.filter_cutoff_hz, lo))
                      + env_value * s.filter_env_amount * math.log(hi / lo))
        return min(max(math.exp(log_cutoff), lo), hi)

    def filter_gains_at(self, freqs_hz: Sequence[float], env_value: float) -> np.ndarray:
        s = self._settings
        cutoff = self._modulated_cutoff_hz(env_value)
        return self._filter.gains(
            np.asarray(freqs_hz, dtype=np.float64), cutoff,
            s.filter_resonance_q, s.filter_resonance_gain_cap,
            s.filter_per_harmonic_min_gain)

    # ── per-chunk evaluation (call exactly once per chunk boundary) ─────────
    def next_chunk_gains(self, f0: float, omegas: Sequence[float],
                         chunk_samples: int) -> np.ndarray:
        s = self._settings
        dt = chunk_samples / float(s.fs_out)
        freqs = np.asarray(omegas, dtype=np.float64) * f0

        if (self._disable_pending and self._amp_env.state == "idle"
                and self._filter_env.state == "idle"):
            self._disable_pending = False
            self._enabled = False
            self._amp_env.reset()
            self._filter_env.reset()

        if not self._enabled:
            amp_gain = 1.0
            filter_env_value = 0.0
        else:
            raw = self._amp_env.advance(dt, s.amp_env_attack_s, s.amp_env_decay_s,
                                        s.amp_env_sustain, s.amp_env_release_s)
            # Floor the held part of the envelope so a low sustain can't
            # starve the decoder; the release tail is allowed to reach true
            # zero (that's ordinary end-of-signal silence to the decoder).
            if self._amp_env.state in _ACTIVE_STATES:
                amp_gain = max(raw, s.envelope_min_gain)
            else:
                amp_gain = raw
            filter_env_value = self._filter_env.advance(
                dt, s.filter_env_attack_s, s.filter_env_decay_s,
                s.filter_env_sustain, s.filter_env_release_s)

        filter_gains = self.filter_gains_at(freqs, filter_env_value)
        self._last_filter_gains = filter_gains
        self._last_filter_env_value = filter_env_value
        return filter_gains * amp_gain

    # ── display support (post-filter wave viewer) ───────────────────────────
    def display_filter_gains(self, freqs_hz: Sequence[float]) -> np.ndarray:
        """Filter-only gain curve at the most recent envelope position
        (amp envelope excluded so the displayed *shape* doesn't shrink with
        volume). Computed fresh so it tracks slider moves while idle."""
        return self.filter_gains_at(freqs_hz, self._last_filter_env_value)
