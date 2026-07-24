import math

from synthflow.Payload.SymbolRow import SymbolRow


def test_empty_offsets_returns_empty():
    row = SymbolRow([])
    assert row.resample_to_size(5) == []


def test_num_samples_non_positive_returns_empty():
    row = SymbolRow([1.0, 2.0, 3.0])
    assert row.resample_to_size(0) == []
    assert row.resample_to_size(-3) == []


def test_single_element_broadcasts():
    row = SymbolRow([7.0])
    result = row.resample_to_size(4)
    assert result == [7.0, 7.0, 7.0, 7.0]


def test_same_size_is_identity_copy_not_same_object():
    offsets = [1.0, 2.0, 3.0]
    row = SymbolRow(offsets)
    result = row.resample_to_size(3)
    assert result == offsets
    assert result is not offsets


def test_upsampling_linear_interpolation():
    row = SymbolRow([0.0, 10.0])
    result = row.resample_to_size(3)
    assert math.isclose(result[0], 0.0)
    assert math.isclose(result[1], 5.0)
    assert math.isclose(result[2], 10.0)
