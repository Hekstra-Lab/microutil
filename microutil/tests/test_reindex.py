import numpy as np
import xarray as xr

from microutil.track_utils import reindex

from .utils import load_zarr

input = np.array(
    [
        [[[0.0, 1.0], [1.0, 3.0]], [[0.0, 1.0], [2.0, 9.0]]],
        [[[0.0, 1.0], [1.0, 4.0]], [[1.0, 0.0], [2.0, 8.0]]],
    ]
)


expected = np.array(
    [
        [[[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [2.0, 3.0]]],
        [[[0.0, 1.0], [1.0, 2.0]], [[1.0, 0.0], [2.0, 3.0]]],
    ]
)


def test_reindex_numpy():
    assert np.allclose(expected, reindex(input, inplace=False))


def test_reindex_xarray():
    input_xr = xr.DataArray(input, dims=['t', 'p', 'y', 'x'])
    assert np.allclose(expected, reindex(input_xr, inplace=False))


def test_reindex_area():
    input = load_zarr('test-data/reindex/input')
    expected = load_zarr('test-data/reindex/expected')
    assert np.allclose(expected, reindex(input, inplace=False, min_area=200))
