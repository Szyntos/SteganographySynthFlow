"""Audio engines for the rack GUI.

Each engine owns a sounddevice stream and drives EncoderDSP / DecoderDSP.
All strategy/payload/codec/sink assembly stays inside the DSP classes; the
engines only handle streaming, device selection, note gating and thread
safety. No tkinter in this module.

Three flavours, matching the rack's module combinations:
  EncoderEngine — encoder module alone (output stream)
  DecoderEngine — decoder module alone (duplex stream: line-in → decode → out)
  LinkedEngine  — both modules: encoder feeds the decoder internally, one
                  output stream, shared transmission parameters kept in sync
"""

import collections
import threading
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from audio_callback_diag import log_callback_event
from DecoderDSP import DecoderDSP
from EncoderDSP import EncoderDSP
from MidiDeviceInput import MidiDeviceInput, list_midi_input_devices
from MidiFilePlayer import MidiFilePlayer
from NoteState import NoteState, midi_note_to_hz
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import SinkBehaviour
from WaveParams import WaveParams


def list_audio_devices(kind: str) -> List[Tuple[int, str]]:
    """kind: 'input' or 'output'. Returns (index, display-name) pairs."""
    if sd is None:
        return []
    key = f"max_{kind}_channels"
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev[key] > 0:
            devices.append((idx, f"[{idx}] {dev['name']}"))
    return devices


def _require_sounddevice() -> None:
    if sd is None:
        raise RuntimeError("sounddevice is not installed.\nRun: pip install sounddevice")


class _NoteControl:
    """MIDI device / QWERTY / on-screen piano / MIDI file note sources feeding
    one monophonic NoteState, plus the gate flags the audio callback reads."""

    def __init__(self):
        self.note_state = NoteState()
        self.midi_input = MidiDeviceInput(self.note_state)
        self.midi_file_player = MidiFilePlayer(self.note_state)
        self.midi_enabled = False
        self.keyboard_enabled = False
        self.pointer_active = False

    def gate_active(self) -> bool:
        return (self.midi_enabled or self.keyboard_enabled
                or self.pointer_active or self.midi_file_player.is_playing())

    def shutdown(self) -> None:
        self.midi_file_player.stop()
        self.midi_input.stop()
        self.midi_enabled = False


class _EngineBase:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._volume: float = 1.0
        self._stream = None
        self._lock = threading.Lock()

    def set_volume(self, vol: float) -> None:
        self._volume = float(vol)

    def is_running(self) -> bool:
        return self._stream is not None

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def shutdown(self) -> None:
        self.stop()


class _EncoderSideMixin:
    """Encoder-facing controls shared by EncoderEngine and LinkedEngine.

    Expects self._enc (EncoderDSP), self._lock, self._notes (_NoteControl).
    """

    def set_output_device(self, device: Optional[int]) -> None:
        self._output_device = device

    def load_payload_file(self, file_path: str) -> None:
        with self._lock:
            self._enc.load_payload_file(file_path)

    def get_payload_path(self, kind: Optional[str] = None) -> Optional[str]:
        with self._lock:
            return self._enc.get_payload_path(kind)

    def get_position_fraction(self) -> float:
        with self._lock:
            return self._enc.get_position_fraction()

    def set_position_fraction(self, fraction: float) -> None:
        with self._lock:
            self._enc.set_position_fraction(fraction)

    def get_encoder_f0(self) -> float:
        return self._enc.get_f0()

    def get_wave_params(self) -> WaveParams:
        with self._lock:
            return self._enc.get_wave_params()

    def set_wave_params(self, params: WaveParams) -> None:
        with self._lock:
            self._enc.set_wave_params(params)

    def get_filtered_wave_params(self) -> WaveParams:
        with self._lock:
            return self._enc.get_filtered_wave_params()

    def _drive_voice(self) -> bool:
        """Run inside the audio callback (lock held): forward the note gate to
        the synth voice as envelope edges. Returns True while a note is held.
        Envelope on/off replaces the old hard output muting — release tails
        ring out after note-off, classic monosynth style."""
        gate_active = self._notes.gate_active()
        self._enc.set_voice_enabled(gate_active)
        midi_note = self._notes.note_state.current_note_or(-1) if gate_active else -1
        note_held = midi_note >= 0
        if gate_active and midi_note != self._last_gate_note:
            # Retrigger on every key change — legato included: each new key
            # resets both envelopes and restarts the attack from zero.
            if note_held:
                self._enc.note_on()
            else:
                self._enc.note_off()
        self._last_gate_note = midi_note if gate_active else None
        return note_held

    # ── note control ────────────────────────────────────────────────────────
    def list_midi_devices(self) -> List[str]:
        return list_midi_input_devices()

    def get_note_state(self) -> NoteState:
        return self._notes.note_state

    def get_midi_file_player(self) -> MidiFilePlayer:
        return self._notes.midi_file_player

    def get_active_note(self) -> Optional[int]:
        note = self._notes.note_state.current_note_or(-1)
        return note if note >= 0 else None

    def set_midi_enabled(self, enabled: bool, device_name: Optional[str] = None) -> None:
        with self._lock:
            if enabled:
                self._notes.midi_input.start(device_name)
            else:
                self._notes.midi_input.stop()
            self._notes.midi_enabled = enabled

    def set_keyboard_enabled(self, enabled: bool) -> None:
        self._notes.keyboard_enabled = bool(enabled)

    def set_pointer_active(self, active: bool) -> None:
        self._notes.pointer_active = bool(active)


class _DecoderSideMixin:
    """Decoder-facing controls shared by DecoderEngine and LinkedEngine.

    Expects self._dec (DecoderDSP) and self._lock.
    """

    def set_sink_behaviour(self, behaviour: SinkBehaviour) -> None:
        with self._lock:
            self._dec.set_sink_behaviour(behaviour)

    def set_harmonic_scalars(self, scalars) -> None:
        with self._lock:
            self._dec.set_harmonic_scalars(scalars)

    def set_on_image(self, cb: Optional[Callable]) -> None:
        with self._lock:
            self._dec.set_on_image(cb)

    def set_on_raw_image(self, cb: Optional[Callable]) -> None:
        with self._lock:
            self._dec.set_on_raw_image(cb)

    def set_on_data(self, cb: Optional[Callable]) -> None:
        with self._lock:
            self._dec.set_on_data(cb)

    def set_on_raw_data(self, cb: Optional[Callable]) -> None:
        with self._lock:
            self._dec.set_on_raw_data(cb)

    def set_on_text(self, cb: Optional[Callable]) -> None:
        with self._lock:
            self._dec.set_on_text(cb)

    def set_on_raw_text(self, cb: Optional[Callable]) -> None:
        with self._lock:
            self._dec.set_on_raw_text(cb)

    def get_latest_text(self) -> Optional[str]:
        with self._lock:
            return self._dec.get_latest_text()

    def dump_decoded_audio_to_wav(self, file_path: str) -> None:
        with self._lock:
            self._dec.dump_decoded_audio_to_wav(file_path)

    def dump_decoded_bytes_to_file(self, file_path: str) -> None:
        with self._lock:
            self._dec.dump_decoded_bytes_to_file(file_path)

    def set_f0_estimator_mode(self, mode: str) -> None:
        with self._lock:
            self._dec.set_f0_estimator_mode(mode)

    def set_pitch_quantize(self, enabled: bool) -> None:
        self._dec.set_pitch_quantize(enabled)

    def set_resample_method(self, method: str) -> None:
        with self._lock:
            self._dec.set_resample_method(method)

    def set_decoder_f0(self, f0: float) -> None:
        with self._lock:
            self._dec.set_f0(f0)

    def set_tune_offset(self, offset: int) -> None:
        with self._lock:
            self._dec.set_tune_offset(offset)

    def get_tune_offset(self) -> int:
        return self._dec.get_tune_offset()

    def get_estimated_f0(self) -> float:
        return self._dec.get_estimated_f0()

    def get_confidence(self) -> float:
        return self._dec.get_confidence()

    def get_drop_run(self) -> int:
        return self._dec.get_drop_run()

    def is_gated(self) -> bool:
        return self._dec.is_gated()


class EncoderEngine(_EngineBase, _EncoderSideMixin):
    """Standalone encoder: EncoderDSP → selected output device."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._output_device: Optional[int] = None
        self._output_device2: Optional[int] = None
        self._output2_enabled: bool = False
        self._stream2 = None
        self._secondary_buffer = collections.deque(maxlen=4)
        self._enc = EncoderDSP(settings)
        self._enc.set_f0(settings.pitch_default_hz)
        self._notes = _NoteControl()
        self._last_gate_note: Optional[int] = None

    def set_output_device2(self, device: Optional[int]) -> None:
        self._output_device2 = device

    def set_output2_enabled(self, enabled: bool) -> None:
        self._output2_enabled = bool(enabled)

    def set_strategy_kind(self, kind: str) -> None:
        with self._lock:
            self._enc.set_strategy_kind(kind)

    def set_payload_kind(self, kind: str) -> None:
        with self._lock:
            self._enc.set_payload_kind(kind)

    def set_codec_mode(self, mode: SerializerMode) -> None:
        with self._lock:
            self._enc.set_codec_mode(mode)

    def set_bits_per_symbol(self, bits: int) -> None:
        with self._lock:
            self._enc.set_bits_per_symbol(bits)

    def set_audio_samples_per_symbol(self, samples_per_symbol: int) -> None:
        with self._lock:
            self._enc.set_audio_samples_per_symbol(samples_per_symbol)

    def set_encoder_f0(self, f0: float) -> None:
        self._enc.set_f0(f0)

    def apply_dsp_settings(self, values: dict) -> None:
        with self._lock:
            self._enc.apply_dsp_settings(values)

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        _cb_start = time.perf_counter()
        try:
            with self._lock:
                # Pitch follows the held note; the encoder keeps advancing
                # through rests (the amp envelope shapes gaps now), so payload
                # position and phases stay continuous.
                note_held = self._drive_voice()
                if note_held:
                    self._enc.set_f0(
                        midi_note_to_hz(self._notes.note_state.current_note_or(-1)))

                enc_chunk = self._enc.process(frames)
                arr = np.array(enc_chunk.get_samples(), dtype=np.float32) * self._volume
                outdata[:, 0] = arr
                if self._output2_enabled:
                    self._secondary_buffer.append(arr.copy())
        finally:
            duration = time.perf_counter() - _cb_start
            budget = frames / float(self._settings.fs_out)
            log_callback_event("encoder", status, duration, budget)

    def _callback2(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        # Mirrors whatever the primary stream just produced; on underrun
        # (secondary polled faster than the primary fills it) plays silence
        # rather than blocking the audio thread.
        if self._secondary_buffer:
            block = self._secondary_buffer.popleft()
            if len(block) == frames:
                outdata[:, 0] = block
                return
        outdata.fill(0)

    def start(self) -> None:
        _require_sounddevice()
        self._stream = sd.OutputStream(
            samplerate=self._settings.fs_out,
            blocksize=self._settings.audio_driver_polling_rate,
            channels=1,
            dtype="float32",
            device=self._output_device,
            callback=self._callback,
        )
        self._stream.start()
        if self._output2_enabled:
            self._secondary_buffer.clear()
            self._stream2 = sd.OutputStream(
                samplerate=self._settings.fs_out,
                blocksize=self._settings.audio_driver_polling_rate,
                channels=1,
                dtype="float32",
                device=self._output_device2,
                callback=self._callback2,
            )
            self._stream2.start()

    def stop(self) -> None:
        super().stop()
        if self._stream2 is not None:
            self._stream2.stop()
            self._stream2.close()
            self._stream2 = None

    def shutdown(self) -> None:
        self._notes.shutdown()
        self.stop()


class DecoderEngine(_EngineBase, _DecoderSideMixin):
    """Standalone decoder: selected input device → DecoderDSP → output device."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._input_device: Optional[int] = None
        self._output_device: Optional[int] = None
        self._dec = DecoderDSP(settings)

    def set_input_device(self, device: Optional[int]) -> None:
        self._input_device = device

    def set_output_device(self, device: Optional[int]) -> None:
        self._output_device = device

    def set_strategy_kind(self, kind: str) -> None:
        with self._lock:
            self._dec.set_strategy_kind(kind)

    def set_payload_kind(self, kind: str) -> None:
        with self._lock:
            self._dec.set_payload_kind(kind)

    def set_codec_mode(self, mode: SerializerMode) -> None:
        with self._lock:
            self._dec.set_codec_mode(mode)

    def set_bits_per_symbol(self, bits: int) -> None:
        with self._lock:
            self._dec.set_bits_per_symbol(bits)

    def set_audio_samples_per_symbol(self, samples_per_symbol: int) -> None:
        with self._lock:
            self._dec.set_audio_samples_per_symbol(samples_per_symbol)

    def apply_dsp_settings(self, values: dict) -> None:
        with self._lock:
            self._dec.apply_dsp_settings(values)

    def _callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int,
                  time_info, status) -> None:
        _cb_start = time.perf_counter()
        try:
            with self._lock:
                samples = indata[:, 0].astype(np.float32)
                dec_samples = self._dec.process_chunk(samples, frames)
                outdata[:, 0] = np.array(dec_samples, dtype=np.float32) * self._volume
        finally:
            duration = time.perf_counter() - _cb_start
            budget = frames / float(self._settings.fs_out)
            log_callback_event("decoder", status, duration, budget)

    def start(self) -> None:
        _require_sounddevice()
        self._dec.reset()
        self._stream = sd.Stream(
            samplerate=self._settings.fs_out,
            blocksize=self._settings.audio_driver_polling_rate,
            channels=1,
            dtype="float32",
            device=(self._input_device, self._output_device),
            callback=self._callback,
        )
        self._stream.start()


class LinkedEngine(_EngineBase, _EncoderSideMixin, _DecoderSideMixin):
    """Encoder and decoder in one stream: the encoder's samples are fed
    straight into the decoder, and the monitor source picks which of the two
    reaches the speakers. Shared transmission knobs are applied to both DSPs
    so they can never drift apart."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._output_device: Optional[int] = None
        self._source: str = "encoder"
        self._enc_gain: float = 1.0
        self._dec_gain: float = 1.0
        self._enc = EncoderDSP(settings)
        self._dec = DecoderDSP(settings)
        self._enc.set_f0(settings.pitch_default_hz)
        self._dec.set_f0(settings.pitch_default_hz)
        self._notes = _NoteControl()
        self._last_gate_note: Optional[int] = None

    def set_source(self, source: str) -> None:
        self._source = source

    def get_source(self) -> str:
        return self._source

    def set_enc_gain(self, gain: float) -> None:
        self._enc_gain = float(gain)

    def set_dec_gain(self, gain: float) -> None:
        self._dec_gain = float(gain)

    # ── shared transmission parameters (both DSPs) ─────────────────────────
    def set_strategy_kind(self, kind: str) -> None:
        with self._lock:
            self._enc.set_strategy_kind(kind)
            self._dec.set_strategy_kind(kind)

    def set_payload_kind(self, kind: str) -> None:
        with self._lock:
            self._enc.set_payload_kind(kind)
            self._dec.set_payload_kind(kind)

    def set_codec_mode(self, mode: SerializerMode) -> None:
        with self._lock:
            self._enc.set_codec_mode(mode)
            self._dec.set_codec_mode(mode)

    def set_bits_per_symbol(self, bits: int) -> None:
        with self._lock:
            self._enc.set_bits_per_symbol(bits)
            self._dec.set_bits_per_symbol(bits)

    def set_audio_samples_per_symbol(self, samples_per_symbol: int) -> None:
        with self._lock:
            self._enc.set_audio_samples_per_symbol(samples_per_symbol)
            self._dec.set_audio_samples_per_symbol(samples_per_symbol)

    def apply_dsp_settings(self, values: dict) -> None:
        # Both DSPs share one Settings object: mutate it once through the
        # encoder facade, then give the decoder its structural rebuild.
        with self._lock:
            self._enc.apply_dsp_settings(values)
            self._dec.rebuild_after_settings_change()

    def set_encoder_f0(self, f0: float) -> None:
        self._enc.set_f0(f0)

    def set_wave_params(self, params: WaveParams) -> None:
        # Linked: the decoder's projections must chase the encoder's omegas.
        with self._lock:
            self._enc.set_wave_params(params)
            self._dec.set_harmonic_scalars(params.omegas)

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        _cb_start = time.perf_counter()
        try:
            with self._lock:
                # Pitch follows the held note on both DSPs (so a manual-f0
                # decoder stays in tune); rests are shaped by the amp envelope
                # in the encoded samples, so the decoder hears the same
                # release tails and gaps a real listener would.
                note_held = self._drive_voice()
                if note_held:
                    f0 = midi_note_to_hz(self._notes.note_state.current_note_or(-1))
                    self._enc.set_f0(f0)
                    self._dec.set_f0(f0)

                enc_chunk = self._enc.process(frames)
                enc_samples = np.array(enc_chunk.get_samples(), dtype=np.float32)

                dec_samples = self._dec.process_chunk(enc_samples, frames)

                enc_out = enc_samples * self._enc_gain
                dec_out = np.array(dec_samples, dtype=np.float32) * self._dec_gain
                if self._source == "both":
                    # Averaged, not summed, so hearing both at once doesn't
                    # clip relative to either soloed source.
                    samples = (enc_out + dec_out) * 0.5
                elif self._source == "encoder":
                    samples = enc_out
                else:
                    samples = dec_out
                outdata[:, 0] = np.array(samples, dtype=np.float32) * self._volume
        finally:
            duration = time.perf_counter() - _cb_start
            budget = frames / float(self._settings.fs_out)
            log_callback_event("linked", status, duration, budget)

    def start(self) -> None:
        _require_sounddevice()
        self._dec.reset()
        self._stream = sd.OutputStream(
            samplerate=self._settings.fs_out,
            blocksize=self._settings.audio_driver_polling_rate,
            channels=1,
            dtype="float32",
            device=self._output_device,
            callback=self._callback,
        )
        self._stream.start()

    def shutdown(self) -> None:
        self._notes.shutdown()
        self.stop()
