from Sink.temporal_merge_policy import TemporalMergePolicy


def full_mask(n):
    return bytearray(b"\x01" * n)


class TestSimilarityAndCoverage:
    def test_no_persist_returns_sentinel(self):
        policy = TemporalMergePolicy()
        sim, cov = policy.compute_similarity_and_coverage(bytearray(10), full_mask(10))
        assert sim == -1.0
        assert cov == 1.0

    def test_nothing_arrived_returns_sentinel(self):
        policy = TemporalMergePolicy()
        policy.persist = bytearray(10)
        sim, cov = policy.compute_similarity_and_coverage(bytearray(10), bytearray(10))
        assert sim == -1.0
        assert cov == 0.0

    def test_similarity_over_arrived_bytes_only(self):
        policy = TemporalMergePolicy()
        policy.persist = bytearray([100] * 4)
        canvas = bytearray([100, 100, 200, 0])
        arrived = bytearray([1, 1, 0, 0])  # the differing bytes never arrived
        sim, cov = policy.compute_similarity_and_coverage(canvas, arrived)
        assert sim == 1.0
        assert cov == 0.5

    def test_similarity_value(self):
        policy = TemporalMergePolicy()
        policy.persist = bytearray([0] * 10)
        canvas = bytearray([51] * 10)
        sim, cov = policy.compute_similarity_and_coverage(canvas, full_mask(10))
        assert abs(sim - (1.0 - 51.0 / 255.0)) < 1e-9
        assert cov == 1.0


class TestMerge:
    def test_first_frame_direct_copy(self):
        policy = TemporalMergePolicy()
        canvas = bytearray([10, 20, 30, 40])
        policy.merge(canvas, bytearray([1, 1, 0, 0]))
        assert policy.persist == canvas
        # Must be an independent copy, not an alias.
        canvas[0] = 99
        assert policy.persist[0] == 10

    def test_nothing_arrived_is_noop(self):
        policy = TemporalMergePolicy()
        policy.merge(bytearray([1, 2, 3]), bytearray(3))
        assert policy.persist is None

        policy.persist = bytearray([5, 5, 5])
        policy.merge(bytearray([200, 200, 200]), bytearray(3))
        assert policy.persist == bytearray([5, 5, 5])

    def test_blend_when_similar(self):
        policy = TemporalMergePolicy()
        policy.persist = bytearray([100] * 4)
        canvas = bytearray([200] * 4)  # sim = 1 - 100/255 ~= 0.61 >= 0.51 -> blend
        arrived = bytearray([1, 1, 1, 0])
        policy.merge(canvas, arrived)
        expected = int(0.85 * 100 + 0.15 * 200 + 0.5)  # 115
        assert list(policy.persist) == [expected, expected, expected, 100]

    def test_replace_when_dissimilar_and_covered(self):
        policy = TemporalMergePolicy()
        policy.persist = bytearray([0] * 4)
        canvas = bytearray([255] * 4)  # sim = 0 < 0.51, coverage = 1 >= 0.25
        policy.merge(canvas, full_mask(4))
        assert policy.persist == canvas

    def test_low_coverage_blends_instead_of_replacing(self):
        policy = TemporalMergePolicy()
        policy.persist = bytearray([0] * 10)
        canvas = bytearray([255] * 10)
        arrived = bytearray([1] + [0] * 9)  # coverage 0.1 < 0.25 -> blend path
        policy.merge(canvas, arrived)
        expected = int(0.15 * 255 + 0.5)  # 38
        assert list(policy.persist) == [expected] + [0] * 9

    def test_constructor_constants_respected(self):
        policy = TemporalMergePolicy(blend_n=0.5,
                                     replace_similarity_threshold=0.0,
                                     replace_min_coverage=1.0)
        policy.persist = bytearray([0] * 2)
        canvas = bytearray([100] * 2)
        policy.merge(canvas, full_mask(2))
        # replace threshold 0.0 can never trigger; blend with N=0.5
        assert list(policy.persist) == [50, 50]

    def test_reset(self):
        policy = TemporalMergePolicy()
        policy.merge(bytearray([1]), bytearray([1]))
        assert policy.persist is not None
        policy.reset()
        assert policy.persist is None
