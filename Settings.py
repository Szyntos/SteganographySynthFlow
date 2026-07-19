import math


class DspSettings:
    def __init__(self):
        self.fs_out: int = 48_000

        # base_chunk_size is the strategy-independent reference; chunk_size is
        # derived from it by the active strategy's multiplier. Keeping the base
        # separate stops the multiplier compounding when something reads the
        # already-scaled chunk_size back as if it were the base.
        self.base_chunk_size: int = 480 * 2
        self.chunk_size: int = 480 * 2

        self.total_harmonics: int = 50
        self.data_harmonics: int = 40
        self.data_offset: int = 10

        self.phase_range: float = math.pi / 8

        self.base_amplitude: float = 0.9

        self.bits_per_symbol: int = 2

        self.decoder_strategy_alpha: float = 1.0

        # Encoding strategy chunk-size multipliers (relative to base chunk size)
        self.strategy_chunk_size_multiplier: dict = {"two": 1, "four": 2}

    # Derived values are computed on read so they can never go stale when a
    # primitive (chunk_size, bits_per_symbol, data_harmonics, fs_out) changes.
    @property
    def bits_per_chunk(self) -> int:
        return self.data_harmonics * self.bits_per_symbol

    @property
    def bytes_per_chunk(self) -> int:
        return self.bits_per_chunk // 8

    @property
    def pilot_size(self) -> int:
        return self.chunk_size // 2

    @property
    def data_size(self) -> int:
        return self.chunk_size // 2

    @property
    def MSG_FS(self) -> int:
        return (self.data_harmonics * self.fs_out) // self.chunk_size

    def set_bits_per_symbol(self, bits_per_symbol: int) -> None:
        if not 1 <= bits_per_symbol <= 8:
            raise ValueError("bits_per_symbol must be in [1, 8]")
        self.bits_per_symbol = bits_per_symbol

    def set_chunk_size(self, chunk_size: int) -> None:
        if chunk_size <= 0 or chunk_size % 2 != 0:
            raise ValueError("chunk_size must be a positive even integer")
        self.chunk_size = chunk_size

    def validate(self) -> None:
        if self.chunk_size <= 0 or self.chunk_size % 2 != 0:
            raise ValueError("chunk_size must be positive and even")
        if self.base_chunk_size <= 0 or self.base_chunk_size % 2 != 0:
            raise ValueError("base_chunk_size must be positive and even")
        if self.data_offset + self.data_harmonics > self.total_harmonics:
            raise ValueError("data_offset + data_harmonics exceeds total_harmonics")
        if self.fs_out <= 0:
            raise ValueError("fs_out must be positive")
        if not (0 < self.base_amplitude <= 1):
            raise ValueError("base_amplitude must be in (0, 1]")


class AudioIoSettings:
    def __init__(self):
        self.audio_driver_polling_rate: int = 512
        self.max_driver_block_size: int = 512

    def validate(self) -> None:
        if self.audio_driver_polling_rate <= 0:
            raise ValueError("audio_driver_polling_rate must be positive")
        if self.max_driver_block_size <= 0:
            raise ValueError("max_driver_block_size must be positive")


class DecoderBatchingSettings:
    def __init__(self):
        self.decoder_batch_rows: int = 4
        self.decoder_overlap_rows: int = 20
        self.decoder_lookahead_rows: int = 4

    def validate(self) -> None:
        pass


class SyncSettings:
    def __init__(self):
        self.sync_window: int = 12
        self.sync_min_match: int = 9

        self.sync_fuzzy_max_bit_errors_frac: float = 0.08
        self.sync_data_diff_frac: float = 0.35

        self.sync_msg_start: str = "START_SYNC_START"
        self.sync_msg_end: str = "END_SYNC_END"

    def validate(self) -> None:
        if self.sync_min_match > self.sync_window:
            raise ValueError("sync_min_match must be <= sync_window")
        if not (0 <= self.sync_fuzzy_max_bit_errors_frac <= 1):
            raise ValueError("sync_fuzzy_max_bit_errors_frac must be in [0, 1]")
        if not (0 <= self.sync_data_diff_frac <= 1):
            raise ValueError("sync_data_diff_frac must be in [0, 1]")
        if self.sync_data_diff_frac <= self.sync_fuzzy_max_bit_errors_frac:
            raise ValueError("sync_data_diff_frac must be > sync_fuzzy_max_bit_errors_frac")
        if self.sync_msg_start == self.sync_msg_end:
            raise ValueError("sync_msg_start must differ from sync_msg_end")


class PayloadSettings:
    def __init__(self):
        self.modulator_wav_path: str = r"assets/sinesweep.wav"
        self.image_path: str = "assets/test.png"
        self.image_target_w: int = 60
        self.image_target_h: int = 60
        self.image_channels: int = 3

    def validate(self) -> None:
        if self.image_target_w <= 0:
            raise ValueError("image_target_w must be positive")
        if self.image_target_h <= 0:
            raise ValueError("image_target_h must be positive")
        if self.image_channels <= 0:
            raise ValueError("image_channels must be positive")


class GuiSettings:
    def __init__(self):
        # pitch slider
        self.pitch_min_hz: float = 100.0
        self.pitch_max_hz: float = 2000.0
        self.pitch_default_hz: float = 400.0

        # volume slider
        self.volume_min_db: float = -60.0
        self.volume_max_db: float = 0.0
        self.volume_default_db: float = -40.0

        # sliders / previews / polling
        self.position_slider_max: int = 1000
        self.slider_length_px: int = 280
        self.image_preview_size: int = 200
        self.piano_low_note: int = 60
        self.piano_high_note: int = 88
        self.gui_poll_interval_ms: int = 100
        self.gui_note_poll_interval_ms: int = 50
        self.bits_per_symbol_min: int = 1
        self.bits_per_symbol_max: int = 8

        # MIDI file playback tempo knob (multiplier on the file's own tempo)
        self.midi_tempo_scale_min: float = 0.25
        self.midi_tempo_scale_max: float = 4.0
        self.midi_tempo_scale_default: float = 1.0

        # MIDI file playback transpose knob (semitones)
        self.midi_transpose_min: int = -24
        self.midi_transpose_max: int = 24

    def validate(self) -> None:
        if not (self.bits_per_symbol_min <= self.bits_per_symbol_max):
            raise ValueError("bits_per_symbol_min must be <= bits_per_symbol_max")
        if not (self.pitch_min_hz < self.pitch_max_hz):
            raise ValueError("pitch_min_hz must be < pitch_max_hz")
        if not (self.pitch_min_hz <= self.pitch_default_hz <= self.pitch_max_hz):
            raise ValueError("pitch_default_hz must be within [pitch_min_hz, pitch_max_hz]")
        if not (self.volume_min_db < self.volume_max_db):
            raise ValueError("volume_min_db must be < volume_max_db")
        if not (self.volume_min_db <= self.volume_default_db <= self.volume_max_db):
            raise ValueError("volume_default_db must be within [volume_min_db, volume_max_db]")
        if not (self.piano_low_note < self.piano_high_note):
            raise ValueError("piano_low_note must be < piano_high_note")
        if not (0 < self.midi_tempo_scale_min < self.midi_tempo_scale_max):
            raise ValueError("midi_tempo_scale_min must be in (0, midi_tempo_scale_max)")
        if not (self.midi_tempo_scale_min <= self.midi_tempo_scale_default <= self.midi_tempo_scale_max):
            raise ValueError("midi_tempo_scale_default must be within its min/max")
        if not (self.midi_transpose_min <= 0 <= self.midi_transpose_max):
            raise ValueError("midi_transpose range must include 0")


class EnergyGateSettings:
    def __init__(self):
        self.energy_gate_ema_alpha: float = 0.05
        self.energy_gate_abs_floor: float = 1e-6
        self.energy_gate_drop_ratio: float = 0.25

    def validate(self) -> None:
        pass


class DropToleranceSettings:
    def __init__(self):
        # Consecutive missing/gated chunks tolerated (mock silence fed, no
        # reset) before a hard decoder-state reset fires.
        self.drop_tolerance_chunks: int = 3

    def validate(self) -> None:
        if self.drop_tolerance_chunks < 0:
            raise ValueError("drop_tolerance_chunks must be >= 0")


class F0EstimatorSettings:
    def __init__(self):
        self.autocorr_f0_min_hz: float = 200.0
        self.autocorr_f0_max_hz: float = 1200.0
        self.autocorr_rms_floor: float = 1e-4
        self.autocorr_corr_threshold: float = 0.2
        self.fft_f0_n_fft: int = 4096
        self.fft_f0_min_hz: float = 50.0
        self.fft_f0_max_hz: float = 2000.0
        self.fft_rms_floor: float = 1e-6

        self.pitch_quantizer_a4_hz: float = 440.0

    def validate(self) -> None:
        pass


class SinkSettings:
    def __init__(self):
        self.sink_max_buffer_seconds: float = 120.0
        self.raw_binary_sink_max_bytes: int = 512
        self.raw_text_sink_bytes_per_char: int = 4
        self.raw_text_sink_max_chars: int = 200

    def validate(self) -> None:
        pass


class TemporalMergeSettings:
    def __init__(self):
        self.temporal_merge_blend_n: float = 0.85
        self.temporal_merge_replace_similarity_threshold: float = 0.51
        self.temporal_merge_replace_min_coverage: float = 0.25
        self.temporal_merge_similarity_scale: float = 255.0

    def validate(self) -> None:
        pass


class PixelCodecSettings:
    def __init__(self):
        self.pixel_codec_no_data_epsilon: float = 0.03

    def validate(self) -> None:
        pass


class Settings:
    """Composition root over cohesive DSP/GUI/sink/etc. sub-configs.

    Flat attribute access (e.g. ``settings.fs_out``) is preserved via
    delegation to the owning sub-config so existing call sites keep working
    while being migrated incrementally to the grouped form
    (``settings.dsp.fs_out``).
    """

    def __init__(self):
        self.dsp = DspSettings()
        self.audio_io = AudioIoSettings()
        self.decoder_batching = DecoderBatchingSettings()
        self.sync = SyncSettings()
        self.payload = PayloadSettings()
        self.gui = GuiSettings()
        self.energy_gate = EnergyGateSettings()
        self.drop_tolerance = DropToleranceSettings()
        self.f0_estimator = F0EstimatorSettings()
        self.sink = SinkSettings()
        self.temporal_merge = TemporalMergeSettings()
        self.pixel_codec = PixelCodecSettings()
        self._check_group_name_uniqueness()

    def _groups(self):
        return (
            self.dsp, self.audio_io, self.decoder_batching, self.sync, self.payload,
            self.gui, self.energy_gate, self.drop_tolerance, self.f0_estimator,
            self.sink, self.temporal_merge, self.pixel_codec,
        )

    def _check_group_name_uniqueness(self) -> None:
        # Flat delegation below silently picks the first group that has a
        # given name; catch collisions here instead, at construction time.
        seen: dict = {}
        for group in self._groups():
            property_names = [
                name for name, value in vars(group.__class__).items()
                if isinstance(value, property)
            ]
            for name in list(vars(group)) + property_names:
                if name in seen and seen[name] is not group.__class__:
                    raise RuntimeError(
                        f"Settings attribute name collision: '{name}' is defined "
                        f"in both {seen[name].__name__} and {group.__class__.__name__}"
                    )
                seen[name] = group.__class__

    @staticmethod
    def _group_has_own_attr(group, name: str) -> bool:
        # Deliberately not `hasattr`: hasattr() swallows ANY exception raised
        # while evaluating the attribute (not just AttributeError), which
        # would mask a genuine bug in a sub-config property as "not found
        # here, try the next group".
        try:
            object.__getattribute__(group, name)
        except AttributeError:
            return False
        return True

    def __getattr__(self, name: str):
        # Only invoked when normal lookup fails, i.e. for flat legacy names.
        for group in object.__getattribute__(self, "_groups")():
            if self._group_has_own_attr(group, name):
                return getattr(group, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name in Settings._OWN_ATTRS or name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        for group in self._groups():
            if self._group_has_own_attr(group, name):
                setattr(group, name, value)
                return
        # No group owns this name: refuse rather than silently creating a new
        # attribute on Settings, which would shadow nothing today but mask
        # typos and never be read by any consumer.
        raise AttributeError(
            f"Settings has no attribute '{name}'; refusing to create one "
            f"implicitly (set it on the owning sub-config instead)"
        )

    _OWN_ATTRS = frozenset({
        "dsp", "audio_io", "decoder_batching", "sync", "payload", "gui",
        "energy_gate", "drop_tolerance", "f0_estimator", "sink",
        "temporal_merge", "pixel_codec",
    })

    def set_bits_per_symbol(self, bits_per_symbol: int) -> None:
        self.dsp.set_bits_per_symbol(bits_per_symbol)

    def set_chunk_size(self, chunk_size: int) -> None:
        self.dsp.set_chunk_size(chunk_size)

    def validate(self) -> None:
        for group in self._groups():
            group.validate()
