from microutil.tracking import track
import microutil as mu
import numpy as np
from .utils import load_zarr


def test_reindex_area():
    input = load_zarr('test-data/tracking/input')
    expected = load_zarr('test-data/tracking/expected')
    assert np.allclose(expected, track(input))
