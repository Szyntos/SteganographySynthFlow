import math

class Settings:
    def __init__(self):
        self.fs_out: int     = 48_000
        self.chunk_size: int = 480 * 2

        self.audio_driver_polling_rate: int = 512

        self.max_driver_block_size: int = 512

        self.total_harmonics: int = 50
        self.data_harmonics: int  = 49
        self.data_offset: int     = 1

        self.phase_range: float = math.pi / 8

        self.base_amplitude: float = 0.9

        self.bits_per_symbol: int = 2
        self.bits_per_chunk: int  = self.data_harmonics * self.bits_per_symbol
        self.bytes_per_chunk: int = self.bits_per_chunk // 8


        self.pilot_size: int = self.chunk_size // 2
        self.data_size: int  = self.chunk_size // 2


        self.sync_chunk_length: int = 12
        self.sync_window: int       = 12
        self.sync_min_match: int    = 9

        self.sync_fuzzy_max_bit_errors_frac: float = 0.08
        self.sync_data_diff_frac: float            = 0.35

        self.sync_msg_start: str = "START_SYNC_START"
        self.sync_msg_end: str   = "END_SYNC_END"

        self.modulator_wav_path: str = r"assets/sinesweep.wav"
        self.image_path: str = "assets/test.png"
        self.image_target_w: int = 40
        self.image_target_h: int = 40
        self.image_channels: int = 3
        self.image_mode: str = "live"

        self.MSG_FS: int = (self.data_harmonics * self.fs_out) // self.chunk_size

    def set_bits_per_symbol(self, bits_per_symbol: int) -> None:
        if not 1 <= bits_per_symbol <= 8:
            raise ValueError("bits_per_symbol must be in [1, 8]")
        self.bits_per_symbol = bits_per_symbol
        self.bits_per_chunk = self.data_harmonics * bits_per_symbol
        self.bytes_per_chunk = self.bits_per_chunk // 8

    def validate(self):
        if self.chunk_size % 2 != 0:
            print("chunk_size must be even")
        if self.data_offset + self.data_harmonics > self.total_harmonics:
            print("Data harmonics exceed available spectrum")