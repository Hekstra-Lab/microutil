from microutil.segmentation import individualize

from .utils import open_zarr


def test_individualize():
    input = open_zarr('test-data/individualize/input')
    expected = open_zarr('test-data/individualize/expected')
    individualize(input, min_distance=10)
    assert expected.identical(input)
