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
    Compute the radially-binned power spectrum of an image. Assumes xarr has dimensions
    (FOV, Time, z, x, y), essentially different z stacks of BF images.
    Only x and y are explicity acted upon.
    """
    fft_mags = xr.DataArray(
        da.fft.fftshift(da.absolute(da.fft.fft2(xarr.data)) ** 2),
        dims=xarr.dims,
        coords=xarr.coords,
    )
    fft_mags.coords["x"] = np.arange(fft_mags.x.shape[0]) - fft_mags.x.shape[0] / 2
    fft_mags.coords["y"] = np.arange(fft_mags.y.shape[0]) - fft_mags.y.shape[0] / 2
    logR = 0.5 * xr.ufuncs.log1p(fft_mags.coords["x"] ** 2 + fft_mags.coords["y"] ** 2)
    log_power_spectra = xr.ufuncs.log1p(
        fft_mags.groupby_bins(logR, bins=100).mean(dim="stacked_x_y")
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


def squash_zstack(data, squash_fn="max", bf_name="BF", channel_name="channel"):
    """
    Use PLLS to select the best BF slice and compress the fluorescent z stacks using squash_fn.
    Valid squash_fn's are 'max' and 'mean'.
    """
    bf = data.sel({channel_name: bf_name})
    fluo = data.sel(channel=data.channel != bf_name)

    if squash_fn == "max":
        fluo_out = fluo.max("z")
    if squash_fn == "mean":
        fluo_out = fluo.mean("z")

    # Now do PLLS for Brightfield
    power_spec = compute_power_spectrum(bf)
    best_slices = xarray_plls(power_spec, power_spec.group_bins).load().argmin("z")
    best_bf = bf.isel(z=best_slices)

    result = xr.concat((best_bf, fluo_out), dim=channel_name)

    return result.transpose(..., "y", "x")
