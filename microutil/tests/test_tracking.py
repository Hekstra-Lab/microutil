import numpy as np
import xarray as xr

import microutil as mu
from microutil.tracking import track

from .utils import load_zarr, open_zarr


def test_reindex_area():
    input = load_zarr('test-data/tracking/input').astype(int)[None, ...]
    ds = xr.Dataset({'labels': xr.DataArray(input, dims=("S", "T", "Y", "X"))})
    expected = load_zarr('test-data/tracking/expected').astype(int)[None, ...]
    track(ds)
    assert np.allclose(expected, ds['labels'].values)


# don't want to mess with test data so second test
# so we can load in the different way
def test_tracking2():
    input = open_zarr('test-data/tracking2/input').load()
    expected = open_zarr('test-data/tracking2/expected').load()
    mu.track(input)
    assert np.allclose(input['labels'].values, expected['labels'].values)
