import random

from Framing.FrameGenerator import FrameGenerator
from Framing.FramingSyncController import FramingSyncController
from Settings import Settings


def pattern_rows(bits, chunk_len):
    rows = []
    for offset in range(0, len(bits), chunk_len):
        row = bits[offset:offset + chunk_len]
        rows.append(row + [0] * (chunk_len - len(row)))
    return rows


def random_data_row(rng, chunk_len):
    return [rng.randrange(2) for _ in range(chunk_len)]


class TestFramingSyncController:
    def setup_method(self):
        self.settings = Settings()
        self.chunk_len = self.settings.data_harmonics
        frame_generator = FrameGenerator(self.settings)
        self.start_rows = pattern_rows(frame_generator.get_start(), self.chunk_len)
        self.end_rows = pattern_rows(frame_generator.get_end(), self.chunk_len)
        self.controller = FramingSyncController.from_settings(self.settings)

    def test_default_constructed_is_inert(self):
        controller = FramingSyncController()
        assert controller.push([0] * 49) == (False, False, False)
        controller.reset()  # must not raise

    def test_full_frame_sequence(self):
        rng = random.Random(7)
        fires = []
        rows = (self.start_rows
                + [random_data_row(rng, self.chunk_len) for _ in range(20)]
                + self.end_rows)
        for row in rows:
            fires.append(self.controller.push(row))

        n_start = len(self.start_rows)
        # start fires exactly once, on the first data row after the marker
        assert [f[0] for f in fires].count(True) == 1
        assert fires[n_start][0] is True
        # end enters exactly at the last end-marker row
        assert fires[-1][1] is True
        assert all(f[1] is False for f in fires[:-1])
        # is_end_match_now flags exactly the end-marker rows
        assert [f[2] for f in fires] == [False] * (len(rows) - len(self.end_rows)) + [True] * len(self.end_rows)

    def test_fuzzy_tolerance(self):
        rng = random.Random(11)
        corrupted = [row[:] for row in self.start_rows]
        for row in corrupted:
            for index in rng.sample(range(self.chunk_len), 2):  # 2 <= fuzzy_max_errors
                row[index] ^= 1
        for row in corrupted:
            start_fire, _, _ = self.controller.push(row)
            assert start_fire is False
        start_fire, _, _ = self.controller.push(random_data_row(rng, self.chunk_len))
        assert start_fire is True

    def test_reset_rearms(self):
        rng = random.Random(3)
        for row in self.start_rows:
            self.controller.push(row)
        assert self.controller.push(random_data_row(rng, self.chunk_len))[0] is True
        self.controller.reset()
        for row in self.start_rows:
            self.controller.push(row)
        assert self.controller.push(random_data_row(rng, self.chunk_len))[0] is True

    def test_quantize_row_to_bits(self):
        assert FramingSyncController.quantize_row_to_bits([0.0, 1.0, 0.49, 0.51, -1.0]) == [0, 1, 0, 1, 0]
