import os
from pathlib import Path

import dask.array as da

from microutil.loading import load_mm_frames

dir_ = Path(os.path.abspath(__file__)).parent


def test_load_mm_frames():
    """
    For future reference the elements of the images were overwritten to be small 2x2 arrays.
    The elements of each array are:
        [[channel, position],
         [time   ,     z   ]]
    """
    loaded = load_mm_frames(str(dir_.joinpath('test-data/load_mm_frames/input')))
    ref = da.from_zarr(str(dir_.joinpath('test-data/load_mm_frames/reference_file')))
    assert (da.all(loaded.data == ref)).compute()
