import os
from pathlib import Path

import xarray as xr
import zarr

dir_ = Path(os.path.abspath(__file__)).parent


def load_zarr(path):
    """
    Utility to load the file from either dir that pytest might be called from.
    If called from root the path will be different than in the test dir
    """
    return zarr.load(str(dir_.joinpath(path)))


def open_zarr(path):
    """
    Utility to open an xarray dataset from either dir that pytest might be called from.
    If called from root the path will be different than in the test dir
    """
    return xr.open_zarr(str(dir_.joinpath(path)))
