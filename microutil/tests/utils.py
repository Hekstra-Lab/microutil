import zarr
import os
from pathlib import Path

dir_ = Path(os.path.abspath(__file__)).parent


def load_zarr(path):
    """
    Utility to load the file from either dir that pytest might be called from.
    If called from root the path will be different than in the test dir
    """
    return zarr.load(str(dir_.joinpath(path)))
