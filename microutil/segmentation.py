__all__ = [
    "apply_unet",
    "threshold_predictions",
    "individualize",
    "watershed_single_frame_preseeded",
    "peak_mask_to_napari_points",
]


import numpy as np
import scipy.ndimage as ndi
import xarray as xr
from skimage.exposure import equalize_adapthist
from skimage.feature import peak_local_max
from skimage.filters import threshold_isodata
from skimage.morphology import label
from skimage.segmentation import watershed

from .track_utils import _reindex_labels


def apply_unet(data, model, batch_size=None):
    """
    Apply the UNET to make pixel-wise predictions of cell vs not-cell

    Parameters
    ----------
    data : array-like
        The final two axes should be XY
    model : str or unet instance
        Either an instance of a loaded unet model or a path the model weights
    batch_size : int or None default None
        Number of samples per batch for applying neural network. For GPU with
        32G memory and 1020x1024 images batch size can be as large as 10.

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
    row_add = 16 - orig_shape[-2] % 16 if orig_shape[-2] % 16 else 0
    col_add = 16 - orig_shape[-1] % 16 if orig_shape[-1] % 16 else 0

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
    out = model.predict(arr, batch_size=batch_size)[
        ..., : orig_shape[-2], : orig_shape[-1], 0
    ].reshape(orig_shape)
    if is_xarr:
        xarr = xr.DataArray(out, dims=data.dims, coords=data.coords)
        if 'C' in xarr.coords:
            xarr['C'] = 'mask'
        return xarr
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


def napari_points_to_peak_mask(points, shape, S, T):
    """
    Parameters
    ----------
    points : (N, d) array
        The *data* attribute of a napari points layer
    shape : tuple
        The shape of the output mask
    S, T : int

    Returns
    -------
    peak_mask : array of bool
    """
    new_seeds = _process_seeds(points[:, -2:], points[:, :2])[S, T]
    new_seeds = new_seeds[~np.any(np.isnan(new_seeds), axis=1)]
    peak_mask = np.zeros(shape, dtype=np.bool)
    peak_mask[tuple(new_seeds.astype(np.int).T)] = True
    return peak_mask


def _process_seeds(seeds, idxs=None):
    if idxs is None:
        Ss, Ts, Ys, Xs = np.nonzero(seeds.values)
    else:
        idxs = np.asarray(idxs).astype(np.int)
        Ss, Ts = idxs.T
        Ys, Xs = seeds[:, -2], seeds[:, -1]
    # get the maximum number of cells in any frame so we know what to pad to
    # probs could make this part speedier
    max_N = 0
    for s in np.unique(Ss):
        N = np.unique(Ts[Ss == s], return_counts=True)[1].max()
        if N > max_N:
            max_N = N

    T = Ts.max() + 1
    S = Ss.max() + 1
    _seeds = np.zeros([S, T, max_N, 2], np.float32)
    _seeds[...] = np.nan
    for s in range(S):
        s_idx = Ss == s
        for t in range(T):
            t_idx = Ts[s_idx] == t

            _seeds[s, t, : np.sum(t_idx)] = np.hstack(
                [Ys[s_idx][t_idx][:, None], Xs[s_idx][t_idx][:, None]]
            )
    return _seeds


def watershed_single_frame_preseeded(ds, S, T):
    """
    Perform a watershed on a single frame of a dataset. This will
    not populate the watershed labels. They must already exist.
    You probably don't want to use this function when scripting. This is primarily
    provided for usage inside of correct_watershed.

    Parameters
    ----------
    ds : Dataset
    S, T : int
    """
    mask = ds['mask'][S, T]
    topology = -ndi.distance_transform_edt(mask)
    peak_mask = ds['peak_mask'][S, T]
    ds['labels'][S, T] = watershed(topology, label(peak_mask), mask=mask, connectivity=2)


def peak_mask_to_napari_points(peak_mask):
    """
    Convert a peak mask array into the points format that napari expects

    Parameters
    ----------
    peak_mask : (S, T, Y, X) array of bool

    Returns
    -------
    points : (N, 4) array of int
    """
    points = _process_seeds(peak_mask)
    s = points.shape[:-1]
    N = np.cumprod(s)[-1]
    points_transformed = np.hstack(
        [
            a.ravel()[:, None]
            for a in np.meshgrid(*(np.arange(s) for s in points.shape[:-1]), indexing="ij")
        ]
        + [points.reshape((N, 2))]
    )[:, [0, 1, 3, 4]]
    return points_transformed[~np.isnan(points_transformed).any(axis=1)]


def individualize(ds, min_distance=10, connectivity=2, min_area=25):
    """
    Take a dataset and by modifying it inplace turn the mask into individualized
    cell labels and watershed seed points.

    Parameters
    ---------
    ds : (S, T, ..., Y, X) dataset
        Last two dimensions should be XY
    min_distance : int, default: 10
        Passed through to scipy.ndimage.morphology.distance_transform_edt
    connectivity : int, default: 2
        Passed through to skimage.segmentation.watershed
    min_area : number, default: 25
        The minimum number of pixels for an object to be considered a cell.
        If *None* then no cuttoff will be applied, which can reduce computation time.
    """

    def _individualize(mask):
        dtr = ndi.morphology.distance_transform_edt(mask)
        topology = -dtr

        peak_idx = peak_local_max(-topology, min_distance)
        peak_mask = np.zeros_like(mask, dtype=bool)
        peak_mask[tuple(peak_idx.T)] = True

        m_lab = label(peak_mask)

        mask = watershed(topology, m_lab, mask=mask, connectivity=connectivity)
        if min_area is None:
            return mask, peak_mask
        else:
            return _reindex_labels(mask, min_area, inplace=None)[0], peak_mask

    indiv, seeds = xr.apply_ufunc(
        _individualize,
        ds['mask'],
        input_core_dims=[["Y", "X"]],
        output_core_dims=[("Y", "X"), ("Y", "X")],
        dask="parallelized",
        vectorize=True,
    )
    ds['labels'] = indiv
    ds['peak_mask'] = seeds
