import numpy as np
import xarray as xr
import zarr
from microutil.track_utils import reindex
import os
from pathlib import Path

dir_ = Path(os.path.abspath(__file__)).parent


def _load_zarr(path):
    """
    Utility to load the file from either dir that pytest might be called from.
    If called from root the path will be different than in the test dir
    """
    return zarr.load(str(dir_.joinpath(path)))


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
    input = _load_zarr('test-data/reindex/input')
    expected = _load_zarr('test-data/reindex/expected')
    assert np.allclose(expected, reindex(input, inplace=False, min_area=200))
