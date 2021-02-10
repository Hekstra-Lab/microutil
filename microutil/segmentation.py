__all__ = [
    "apply_unet",
    "threshold_predictions",
    "individualize",
]


# these try except blocks are due to the fact that
# tensorflow doesn't support python 3.9 yet (2021-01-29)
# on macOS big sur napari only works on  python 3.9.0
# So two envs are needed for mac users (e.g. indrani)
# this allows importing this file from either env.
import warnings


import numpy as np
import xarray as xr
import dask.array as da
import scipy.ndimage as ndi
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_isodata
from skimage.feature import peak_local_max
from skimage.morphology import label
from skimage.segmentation import watershed
from .track_utils import _reindex_labels


def apply_unet(data, model):
    """
    Apply the UNET to make pixel-wise predictions of cell vs not-cell

    Parameters
    ----------
    data : array-like
        The final two axes should be XY
    model : str or unet instance
        Either an instance of a loaded unet model or a path the model weights

    Returns
    -------
    mask : array-like
        The predicted mask
    """
    from ._unet import unet
    is_xarr = False
    if isinstance(data, xr.DataArray):
        arr = data.values
        is_xarr = True
    else:
        arr = data

    # TODO make this xarray/dask parallelize if applicable
    arr = np.vectorize(equalize_adapthist, signature="(x,y)->(x,y)")(arr)

    orig_shape = arr.shape
    row_add = 16 - orig_shape[-2] % 16
    col_add = 16 - orig_shape[-1] % 16
    npad = [(0, 0)] * arr.ndim
    npad[-2] = (0, row_add)
    npad[-1] = (0, col_add)

    arr = np.pad(arr, npad)

    # manipulate into shape that tensorflow expects
    if len(orig_shape) == 2:
        arr = arr[None, :, :, None]
    else:
        arr = arr.reshape(
            (
                np.prod(orig_shape[:-2]),
                orig_shape[-2] + row_add,
                orig_shape[-1] + col_add,
                1,
            )
        )

    if isinstance(model, str):
        model = unet(model, (None, None, 1))

    # need the final reshape to squeeze off a potential leading 1 in the shape
    # but we can't squeeze because that might remove axis with size 1
    out = model.predict(arr)[..., :-row_add, :-col_add, 0].reshape(orig_shape)
    if is_xarr:
        return xr.DataArray(out, dims=data.dims, coords=data.coords)
    else:
        return out


def threshold_predictions(predictions, threshold=None):
    """
    Parameters
    ----------
    predictions : array-like
    threshold : float or None, default: None
        If None the threshold will automatically determined using
        skimage.filters.threshold_isodata

    Returns
    -------
    mask : array of bool
    """
    if threshold is None:
        threshold = threshold_isodata(np.asarray(predictions))
    return predictions > threshold


def individualize(mask, min_distance=10, connectivity=2, min_area=25):
    """
    Turn a boolean mask into a a mask of cell ids

    Parameters
    ---------
    mask : array-like
        Last two dimensions should be XY
    min_distance : int, default: 10
        Passed through to scipy.ndimage.morphology.distance_transform_edt
    connectivity : int, default: 2
        Passed through to skimage.segmentation.watershed
    min_area : number, default: 25
        The minimum number of pixels for an object to be considered a cell.
        If *None* then no cuttoff will be applied, which can reduce computation time.

    Returns
    -------
    cell_ids : array-like of int
        The mask is now 0 for backgroud and integers for cell ids
    """

    def _individualize(mask):
        dtr = ndi.morphology.distance_transform_edt(mask)
        topology = -dtr

        peak_idx = peak_local_max(-topology, min_distance)
        peak_mask = np.zeros_like(mask, dtype=bool)
        peak_mask[tuple(peak_idx.T)] = True

        m_lab = label(peak_mask)

        mask = watershed(topology, m_lab, mask=mask, connectivity=2)
        if min_area is None:
            return mask
        else:
            return _reindex_labels(mask, min_area, inplace=False)[0]

    return xr.apply_ufunc(
        _individualize,
        mask,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        dask="parallelized",
        vectorize=True,
    )
