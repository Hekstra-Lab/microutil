import dask.array as da
from microutil.loading import load_mm_frames
from .utils import load_zarr
from pathlib import Path
import os

dir_ = Path(os.path.abspath(__file__)).parent

def test_load_mm_frames():
    loaded = load_mm_frames(str(dir_.joinpath('test-data/load_mm_frames/input')))
    ref = da.from_zarr(str(dir_.joinpath('test-data/load_mm_frames/reference_file')))
    assert(da.all(loaded.data==ref)).compute()
