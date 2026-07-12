import math

class Settings:
    def __init__(self):
        self.fs_out: int     = 48_000
        self.chunk_size: int = 480

        self.audio_driver_polling_rate: int = 512

        self.max_driver_block_size: int = 512

        self.total_harmonics: int = 50
        self.data_harmonics: int  = 40
        self.data_offset: int     = 10

        self.phase_range: float = math.pi / 8

        self.base_amplitude: float = 0.9

        self.bits_per_symbol: int = 2
        self.bits_per_chunk: int  = self.data_harmonics * self.bits_per_symbol
        self.bytes_per_chunk: int = self.bits_per_chunk // 8


        self.pilot_size: int = self.chunk_size // 2
        self.data_size: int  = self.chunk_size // 2


        self.sync_window: int       = 12
        self.sync_min_match: int    = 9

        self.sync_fuzzy_max_bit_errors_frac: float = 0.08
        self.sync_data_diff_frac: float            = 0.35

        self.sync_msg_start: str = "START_SYNC_START"
        self.sync_msg_end: str   = "END_SYNC_END"

        self.modulator_wav_path: str = r"assets/sinesweep.wav"
        self.image_path: str = "assets/test.png"
        self.image_target_w: int = 60
        self.image_target_h: int = 60
        self.image_channels: int = 3

        self.MSG_FS: int = (self.data_harmonics * self.fs_out) // self.chunk_size

        # GUI: pitch slider
        self.pitch_min_hz: float = 100.0
        self.pitch_max_hz: float = 2000.0
        self.pitch_default_hz: float = 400.0

        # GUI: volume slider
        self.volume_min_db: float = -60.0
        self.volume_max_db: float = 0.0
        self.volume_default_db: float = -40.0

        # GUI: sliders / previews / polling
        self.position_slider_max: int = 1000
        self.slider_length_px: int = 280
        self.image_preview_size: int = 200
        self.piano_low_note: int = 60
        self.piano_high_note: int = 88
        self.gui_poll_interval_ms: int = 100
        self.gui_note_poll_interval_ms: int = 50
        self.bits_per_symbol_min: int = 1
        self.bits_per_symbol_max: int = 8

        # Decoder resampling batching
        self.decoder_batch_rows: int = 4
        self.decoder_overlap_rows: int = 20
        self.decoder_lookahead_rows: int = 4

        # EnergyGate
        self.energy_gate_ema_alpha: float = 0.05
        self.energy_gate_abs_floor: float = 1e-6
        self.energy_gate_drop_ratio: float = 0.25

        # DropTolerance: consecutive missing/gated chunks tolerated (mock
        # silence fed, no reset) before a hard decoder-state reset fires.
        self.drop_tolerance_chunks: int = 3

        # F0 estimators
        self.autocorr_f0_min_hz: float = 200.0
        self.autocorr_f0_max_hz: float = 1200.0
        self.autocorr_rms_floor: float = 1e-4
        self.autocorr_corr_threshold: float = 0.2
        self.fft_f0_n_fft: int = 4096
        self.fft_f0_min_hz: float = 50.0
        self.fft_f0_max_hz: float = 2000.0
        self.fft_rms_floor: float = 1e-6

        self.pitch_quantizer_a4_hz: float = 440.0

        # Sinks
        self.sink_max_buffer_seconds: float = 120.0
        self.raw_binary_sink_max_bytes: int = 512
        self.raw_text_sink_bytes_per_char: int = 4
        self.raw_text_sink_max_chars: int = 200

        # Temporal merge policy (image reconstruction blending)
        self.temporal_merge_blend_n: float = 0.85
        self.temporal_merge_replace_similarity_threshold: float = 0.51
        self.temporal_merge_replace_min_coverage: float = 0.25
        self.temporal_merge_similarity_scale: float = 255.0

        # Pixel codec
        self.pixel_codec_no_data_epsilon: float = 0.03

        # Encoding strategy weighting
        self.decoder_strategy_alpha: float = 1.0

        # Encoding strategy chunk-size multipliers (relative to base chunk size)
        self.strategy_chunk_size_multiplier: dict = {"two": 1, "four": 2}

    def set_bits_per_symbol(self, bits_per_symbol: int) -> None:
        if not 1 <= bits_per_symbol <= 8:
            raise ValueError("bits_per_symbol must be in [1, 8]")
        self.bits_per_symbol = bits_per_symbol
        self.bits_per_chunk = self.data_harmonics * bits_per_symbol
        self.bytes_per_chunk = self.bits_per_chunk // 8

    def set_chunk_size(self, chunk_size: int) -> None:
        if chunk_size <= 0 or chunk_size % 2 != 0:
            raise ValueError("chunk_size must be a positive even integer")
        self.chunk_size = chunk_size
        self.pilot_size = self.chunk_size // 2
        self.data_size = self.chunk_size // 2
        self.MSG_FS = (self.data_harmonics * self.fs_out) // self.chunk_size

    def validate(self) -> None:
        if self.chunk_size <= 0 or self.chunk_size % 2 != 0:
            raise ValueError("chunk_size must be positive and even")
        if self.data_offset + self.data_harmonics > self.total_harmonics:
            raise ValueError("data_offset + data_harmonics exceeds total_harmonics")
        if not (self.bits_per_symbol_min <= self.bits_per_symbol <= self.bits_per_symbol_max):
            raise ValueError(
                f"bits_per_symbol must be in [{self.bits_per_symbol_min}, {self.bits_per_symbol_max}]"
            )
        if self.fs_out <= 0:
            raise ValueError("fs_out must be positive")
        if self.audio_driver_polling_rate <= 0:
            raise ValueError("audio_driver_polling_rate must be positive")
        if self.max_driver_block_size <= 0:
            raise ValueError("max_driver_block_size must be positive")
        if not (0 < self.base_amplitude <= 1):
            raise ValueError("base_amplitude must be in (0, 1]")
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
        if self.image_target_w <= 0:
            raise ValueError("image_target_w must be positive")
        if self.image_target_h <= 0:
            raise ValueError("image_target_h must be positive")
        if self.image_channels <= 0:
            raise ValueError("image_channels must be positive")
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
        if self.drop_tolerance_chunks < 0:
            raise ValueError("drop_tolerance_chunks must be >= 0")
