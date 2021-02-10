__all__ = [
    "zeros_like",
    "not_xr",
]
import numpy as np
import xarray as xr
import dask.array as da
import zarr


def zeros_like(arr):
    """
    Smooth over the differences zeros_likes for different
    array types

    Parameters
    ----------
    arr : array-like

    Returns
    -------
    An zeroed array of the same array type as *arr*
    """
    if isinstance(arr, np.ndarray):
        return np.zeros_like(arr)
    elif isinstance(arr, xr.DataArray):
        return xr.zeros_like(arr)
    elif isinstance(arr, zarr.Array):
        return zarr.zeros_like(arr)
    elif isinstance(arr, da.Array):
        return da.zeros_like(arr)


def not_xr(arr):
    """
    Make sure an array is not an xarray as that can
    have major implications for indexing
    """
    if isinstance(arr, xr.DataArray):
        return arr.values
    return arr
