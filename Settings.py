import math

class Settings:
    def __init__(self):
        self.fs_out     = 48_000
        self.chunk_size = 480

        self.audio_driver_polling_rate = 512

        self.max_driver_block_size = 512

        self.total_harmonics = 50
        self.data_harmonics  = 40
        self.data_offset     = 1

        self.phase_range = math.pi / 8

        self.base_amplitude = 0.9

        self.bits_per_symbol = 2
        self.bits_per_chunk  = self.data_harmonics * self.bits_per_symbol
        self.bytes_per_chunk = self.bits_per_chunk / 8


        self.pilot_size = self.chunk_size / 2
        self.data_size  = self.chunk_size / 2


        self.sync_chunk_length = 12
        self.sync_window       = 12
        self.sync_min_match    = 9

        self.sync_fuzzy_max_bit_errors_frac = 0.08
        self.sync_data_diff_frac            = 0.35

        self.sync_msg_start = "START_SYNC_START"
        self.sync_msg_end   = "END_SYNC_END"

        self.modulator_wav_path = "assets/idk47.wav"
        self.image_path = "assets/test.png"
        self.image_target_w = 40
        self.image_target_h = 40
        self.image_channels = 3
        self.image_mode = "live"

        self.MSG_FS = (self.data_harmonics * self.fs_out) / self.chunk_size

    def validate(self):
        if self.chunk_size % 2 != 0:
            print("chunk_size must be even")
        if self.data_offset + self.data_harmonics > self.total_harmonics:
            print("Data harmonics exceed available spectrum")