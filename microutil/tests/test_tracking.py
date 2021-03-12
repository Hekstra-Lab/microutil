from microutil.tracking import track
import microutil as mu
import numpy as np
import xarray as xr
from .utils import load_zarr


def test_reindex_area():
    input = load_zarr('test-data/tracking/input').astype(int)[None, ...]
    ds = xr.Dataset({'labels': xr.DataArray(input, dims=("S", "T", "Y", "X"))})
    expected = load_zarr('test-data/tracking/expected').astype(int)[None, ...]
    track(ds)
    assert np.allclose(expected, ds['labels'].values)
