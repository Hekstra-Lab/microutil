import numpy as np
import xarray as xr
import pandas as pd
import dask.array as da

__all__ = [
    "lstsq_slope_dropna",
    "compute_power_spectrum",
    "xarray_plls",
    "squash_zstack",
]


def lstsq_slope_dropna(log_power_spec, logR):
    """
    Compute the slope of a linear fit between a log power specrum and the log
    of the frequency components. This function is mapped over the
    x and y dimensions of an xarray with xarray_plls.

    Parameters
    ----------
    log_power_spec : array-like (N,)
        Log power spectrum values. Dependent variable for the regression.
    logR : array-like (N,)
        Log frequency components. Independent variable for regression.

    Returns
    -------
    slope : np.ndarray (1,)
        Coeffiecent from linear fit.
    """
    # I dont know why there are nans in the groupby but there are nans
    keep = ~np.isnan(log_power_spec)
    A = np.vander(logR[keep], N=2)
    sol = np.linalg.lstsq(A, log_power_spec[keep], rcond=None)
    return np.array([sol[0][0]])


def compute_power_spectrum(xarr, r_bins=100, x_dim='X', y_dim='Y'):
    """
    Compute the radially-binned power spectrum of individual images. Intended use case if
    for xarr to be a set of z stacks of brightfield images.

    Parameters
    ----------
    xarr : xarray.DataArray
        DataArray backed by dask arrays. If the DataArray does not have named dimensions
        "x" and "y" assume that the last two dimensions correspond to image dimensions.
    r_bins : int
        Number of bins to use for radial histogram. Default 100.
    x_dim : str default 'X'
        Name of dimension corresponding to X pixels
    y_dim : str default 'Y'
        Name of dimension corresponding to Y pixels

    Returns
    -------
    log_power_spectrum : (..., r_bins)
        Log power spectrum of each individiual image in xarr.

    """

    if not isinstance(xarr, xr.DataArray):
        raise TypeError("Can only compute power spectra for xarray.DataArrays.")

    if not isinstance(xarr.data, da.Array):
        xarr.data = da.array(xarr.data)

    fft_mags = xr.DataArray(
        da.fft.fftshift(da.absolute(da.fft.fft2(xarr.data)) ** 2),
        dims=xarr.dims,
        coords=xarr.coords,
    )
    fft_mags.coords[x_dim] = np.arange(fft_mags[x_dim].shape[0]) - fft_mags[x_dim].shape[0] / 2
    fft_mags.coords[y_dim] = np.arange(fft_mags[y_dim].shape[0]) - fft_mags[y_dim].shape[0] / 2
    logR = 0.5 * xr.ufuncs.log1p(fft_mags.coords[x_dim] ** 2 + fft_mags.coords[y_dim] ** 2)
    log_power_spectra = xr.ufuncs.log1p(
        fft_mags.groupby_bins(logR, bins=r_bins).mean(dim=f"stacked_{x_dim}_{y_dim}")
    )
    log_power_spectra["group_bins"] = pd.IntervalIndex(
        log_power_spectra.group_bins.values
    ).mid.values

    return log_power_spectra


def xarray_plls(log_power_spec, logR):
    """
    Xarray apply_ufunc wrapper for lstsq_slope_dropna.

    Parameters
    ----------
    log_power_spec : xarray.DataArray
    Log power spectrum values. Dependent variable for the regression.
    logR : xarray.DataArray
    Log frequency components. Should usually be log_power_spec.group_bins.
    Independent variable for regression.

    Returns
    -------
    slope : xarray.DataArray
    Coeffiecents from PLLS libear fit for each frame.
    """
    return xr.apply_ufunc(
        lstsq_slope_dropna,
        log_power_spec,
        logR,
        input_core_dims=[["group_bins"], ["group_bins"]],
        output_core_dims=[[]],
        output_dtypes="float",
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


def squash_zstack(
    data,
    squash_fn="max",
    bf_name="BF",
    channel_name="C",
    x_dim='X',
    y_dim='Y',
    z_dim='Z',
    transpose=True,
):
    """
    Use PLLS to select the best BF slice and compress the fluorescent z stacks using squash_fn.
    Valid squash_fn's are 'max' and 'mean'.
    """

    # If channels are not named, assume data is only BF stacks
    if channel_name is None:
        bf = data

    else:
        bf = data.sel({channel_name: bf_name})
        fluo = data.sel({channel_name: data[channel_name] != bf_name})

        if squash_fn == "max":
            fluo_out = fluo.max(z_dim)
        if squash_fn == "mean":
            fluo_out = fluo.mean(z_dim)

    # Now do PLLS for Brightfield
    power_spec = compute_power_spectrum(bf, x_dim=x_dim, y_dim=y_dim)
    best_slices = xarray_plls(power_spec, power_spec.group_bins).load().argmin(z_dim)
    best_bf = bf.isel({z_dim: best_slices})

    if channel_name is None:
        result = best_bf

    else:
        result = xr.concat((best_bf, fluo_out), dim=channel_name)

    if z_dim in result.dims:
        result = result.drop(z_dim)

    # if transpose:
    #    return result.transpose(..., "y", "x")
    # else:
    #    return result.transpose(..., "x", "y")

    return result.transpose(..., channel_name, y_dim, x_dim)
