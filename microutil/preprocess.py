import dask.array as da
import numpy as np
import xarray as xr
import pandas as pd

__all__ = [
    "lstsq_slope_dropna",
    "compute_power_spectrum",
    "xarray_plls",
    "squash_zstack",
]


def lstsq_slope_dropna(y, logR_array):
    # I dont know why there are nans in the groupby but there are nans
    keep = ~np.isnan(y)
    A = np.vander(logR_array[keep], N=2)
    sol = np.linalg.lstsq(A, y[keep], rcond=None)
    return np.array([sol[0][0]])


def compute_power_spectrum(xarr, r_bins=100):
    """
    Compute the radially-binned power spectrum of individual images. Intended use case if
    for xarr to be a set of z stacks of brightfield images. 

    Parameters
    ----------
    xarr : xarray.DataArray (..., x, y)
    DataArray backed by dask arrays. If the DataArray does not have named dimensions
    "x" and "y" assume that the last two dimensions correspond to image dimensions.

    r_bins : int
    Number of bins to use for radial histogram. Default 100.

    Returns
    -------
    log_power_spectrum : (..., r_bins)
    Log power spectrum of each individiual image in xarr.

    """
    
    if not isinstance(xarr, xr.DataArray) and not isinstance(xarr.data, dask.core.Array):
       raise TypeError("Can only compute power spectra for xarray.DataArrays backed by dask arrays.")

    fft_mags = xr.DataArray(
        da.fft.fftshift(da.absolute(da.fft.fft2(xarr.data)) ** 2),
        dims=xarr.dims,
        coords=xarr.coords,
    )
    fft_mags.coords["x"] = np.arange(fft_mags.x.shape[0]) - fft_mags.x.shape[0] / 2
    fft_mags.coords["y"] = np.arange(fft_mags.y.shape[0]) - fft_mags.y.shape[0] / 2
    logR = 0.5 * xr.ufuncs.log1p(fft_mags.coords["x"] ** 2 + fft_mags.coords["y"] ** 2)
    log_power_spectra = xr.ufuncs.log1p(
        fft_mags.groupby_bins(logR, bins=r_bins).mean(dim="stacked_x_y")
    )
    log_power_spectra["group_bins"] = pd.IntervalIndex(
        log_power_spectra.group_bins.values
    ).mid.values

    return log_power_spectra


def xarray_plls(y, logR_array):
    return xr.apply_ufunc(
        lstsq_slope_dropna,
        y,
        logR_array,
        input_core_dims=[["group_bins"], ["group_bins"]],
        output_core_dims=[[]],
        output_dtypes="float",
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

#def numpy_plls(y, logR_array):
    


def squash_zstack(data, squash_fn="max", bf_name="BF", channel_name="channel", transpose=True):
    """
    Use PLLS to select the best BF slice and compress the fluorescent z stacks using squash_fn.
    Valid squash_fn's are 'max' and 'mean'.
    """
    bf = data.sel({channel_name: bf_name})
    fluo = data.sel({channel_name: data[channel_name] != bf_name})

    if squash_fn == "max":
        fluo_out = fluo.max("z")
    if squash_fn == "mean":
        fluo_out = fluo.mean("z")

    # Now do PLLS for Brightfield
    power_spec = compute_power_spectrum(bf)
    best_slices = xarray_plls(power_spec, power_spec.group_bins).load().argmin("z")
    best_bf = bf.isel(z=best_slices)

    result = xr.concat((best_bf, fluo_out), dim=channel_name)

    if transpose:
        return result.transpose(..., "y", "x").drop("z")
    else:
        return result.transpose(..., "x", "y").drop("z")
