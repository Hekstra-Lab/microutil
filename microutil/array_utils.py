__all__ = [
    "zeros_like",
    "not_xr",
    "axis2int",
]
import dask.array as da
import numpy as np
import xarray as xr
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


def axis2int(arr, axis, fallthrough=None):
    """
    Get the integer index of an axis for xarray or numpy.
    dims contain *axis*.

    Parameters
    ----------
    arr : ndarry or xarray
    axis : int, str
        The axis to find the index of
    fallthrough : int or None, default: None
        The value to return if the inference doesn't work.

    Returns
    -------
    i : int
        The index in dims.

    """
    if isinstance(axis, int):
        return axis
    else:
        if isinstance(arr, xr.DataArray) and isinstance(axis, str) and (axis in arr.dims):
            return arr.dims.index(axis)
    return fallthrough
