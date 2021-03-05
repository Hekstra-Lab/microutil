from microutil.segmentation import individualize
import microutil as mu
import numpy as np
import xarray as xr
from .utils import open_zarr


def test_individualize():
    input = open_zarr('test-data/individualize/input')
    expected = open_zarr('test-data/individualize/expected')
    individualize(input)
    assert expected.identical(input)
