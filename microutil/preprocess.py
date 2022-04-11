import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from fast_histogram import histogram1d

__all__ = [
    "mode_norm",
    "normalize_fluo",
    "lstsq_slope_dropna",
    "compute_power_spectrum",
    "xarray_plls",
    "squash_zstack",
]


def mode_norm(arr, n_bins=4096, eps=0.01):
    """
    Normalize a single image (arr) by subtracting the mode, zeroing elements
    less than 0 after subtraction, and dividing by the maximum pixel value.

    Parameters
    ----------
    arr : np.array
        Single image to normalize.
    n_bins : int default 4096
        Number of bins in the histogram used to compute the most common
        pixel value. Default 4096 works well for ~1Mb images.
    eps : float default 0.01
        Small offset to expand the edge bins in the histogram by in order
        to ensure that the min and max pixels are properly included.

    Returns
    -------
    normed : np.array (same shape as arr)
        Image with values normalized to be in the range [0,1]
    """

    hist_range = (np.min(arr) - eps, np.max(arr) + eps)
    w = (hist_range[1] - hist_range[0]) / n_bins
    mode_idx = np.argmax(histogram1d(arr, bins=n_bins, range=hist_range))
    mode_val = hist_range[0] + w * (mode_idx + 0.5)
    normed = np.clip((arr - mode_val) / (hist_range[1] - mode_val), 0, 1)
    return normed


def normalize_fluo(imgs, n_bins=4096, eps=0.01, dims=list('STCZYX')):
    """
    Normalize all images in a DataArray by applying mode_norm independently
    to each frame.

    Parameters
    ----------
    imgs : xr.DataArray
        DataArray containing fluorescence images. Can have any number of dims
        but must have dims corresponding to the 2D spatial dimensions Y and X.
    n_bins : int default 4096
        Number of bins in the histogram used to compute the most common
        pixel value. Default 4096 works well for ~1Mb images.
    eps : float default 0.01
        Small offset to expand the edge bins in the histogram by in order
        to ensure that the min and max pixels are properly included.
    dims : list of str default list('STCZYX')
        Dimensions names used in imgs. Only Y and X are explicitly used here.

    Returns
    -------
    normed : xr.DataArray same shape as imgs
        Images with values normalized to be in the range [0,1]
    """
    S, T, C, Z, Y, X = dims
    return xr.apply_ufunc(
        mode_norm,
        imgs,
        kwargs={'n_bins': n_bins, 'eps': eps},
        input_core_dims=[[Y, X]],
        output_core_dims=[[Y, X]],
        vectorize=True,
        dask='parallelized',
    )


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
    channel_dim="C",
    z_dim='Z',
    y_dim='Y',
    x_dim='X',
):
    """
    Use PLLS to select the best BF slice and compress the fluorescent z stacks using squash_fn.

    Parameters
    ----------
    data : xarray.DataArray
    squash_fn : str default 'mean'
    bf_name : str default 'BF'
    channel_dim : str default 'C'
    z_dim : str default 'Z'
    y_dim : str default 'Y'
    x_dim : str default 'X'

    Returns
    -------
    squshed : xarray.DataArray
        Data array with the Z dimension squashed. Dims will be in STCYX order. Dims
        other than z_dim will be the same as input DataAray.
    """

    # If channels are not named, assume data is only BF stacks
    if channel_dim is None:
        bf = data

    else:
        bf = data.sel({channel_dim: bf_name})
        fluo = data.sel({channel_dim: data[channel_dim] != bf_name})

        if squash_fn == "max":
            fluo_out = fluo.max(z_dim)
        if squash_fn == "mean":
            fluo_out = fluo.mean(z_dim)

    # Now do PLLS for Brightfield
    power_spec = compute_power_spectrum(bf, x_dim=x_dim, y_dim=y_dim)
    best_slices = xarray_plls(power_spec, power_spec.group_bins).load().argmin(z_dim)
    best_bf = bf.isel({z_dim: best_slices})

    if channel_dim is None:
        result = best_bf

    else:
        result = xr.concat((best_bf, fluo_out), dim=channel_dim)

    if z_dim in result.dims:
        result = result.drop(z_dim)

    return result.transpose(..., channel_dim, y_dim, x_dim)
