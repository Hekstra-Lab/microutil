__all__ = [
    "apply_unet",
    "threshold_predictions",
    "individualize",
    "watershed_single_frame_preseeded",
    "peak_mask_to_napari_points",
    "fast_otsu",
    "calc_thresholds",
    "relabel_dt",
    "relabel_fluo",
    "relabel_product",
    "merge_overlaps_sequential",
    "merge_overlaps_timeseries",
    "merge_overlaps",
    "labels_to_zarr",
]


import warnings

import numpy as np
import scipy.ndimage as ndi
import xarray as xr
import zarr
from fast_histogram import histogram1d
from fast_overlap import overlap
from skimage.exposure import equalize_adapthist
from skimage.feature import peak_local_max
from skimage.filters import threshold_isodata, threshold_otsu
from skimage.morphology import label, remove_small_holes
from skimage.segmentation import relabel_sequential, watershed

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


def individualize(
    ds,
    topology=None,
    min_distance=3,
    connectivity=2,
    min_area=25,
    threshold=None,
    dims=list('STCZYX'),
):
    """
    Take a dataset and by modifying it inplace turn the mask into individualized
    cell labels and watershed seed points.

    Parameters
    ---------
    ds : (S, T, ..., Y, X) dataset
        Last two dimensions should be XY
    topology: xr.DatArray, str, or None, default None
        Optional topolgoy to use for watershed. If None,
        use the negative of the distance transform. If str,
        it will be assumed to be a variable in ds. Datarrays
        will be used directly by apply_ufunc and must have
        Y and X as dims.
    min_distance : int, default: 10
        Passed through to scipy.ndimage.morphology.distance_transform_edt
    connectivity : int, default: 2
        Passed through to skimage.segmentation.watershed
    min_area : number, default: 25
        The minimum number of pixels for an object to be considered a cell.
        If *None* then no cuttoff will be applied, which can reduce computation time.
    threshold: float or None default None
        Threshold to be passed to peak_local_max.
    """

    S, T, C, Z, Y, X = dims

    if isinstance(topology, str):
        use_topology = ds[topology]
        topo_core_dims = [Y, X]
    elif isinstance(topology, xr.DataArray):
        use_topology = topology
        topo_core_dims = [Y, X]
    else:
        use_topology = None
        topo_core_dims = []

    if topology is not None and threshold is None:
        raise ValueError(
            "Must supply a threshold array which matches non-core dims of topology."
            "Consider using mu.segmention.fast_otsu on -topology or simply"
            "topology.max(['Y','X']). Failure to pass an array here causes dask"
            "to hang for reasons that I do not fully understand."
        )

    def _individualize(mask, topology, threshold):
        if topology is None:
            topology = -ndi.morphology.distance_transform_edt(mask)

        peak_idx = peak_local_max(-topology, min_distance, threshold_abs=threshold)
        peak_mask = np.zeros_like(mask, dtype=bool)
        peak_mask[tuple(peak_idx.T)] = True

        labelled_peaks = label(peak_mask)
        mask = watershed(topology, labelled_peaks, mask=mask, connectivity=connectivity)
        mask = relabel_sequential(mask)[0]
        if min_area is None:
            return mask, peak_mask
        else:
            return _reindex_labels(mask, min_area, inplace=None)[0], peak_mask

    indiv, seeds = xr.apply_ufunc(
        _individualize,
        ds['mask'],
        use_topology,
        threshold,
        input_core_dims=[[Y, X], topo_core_dims, []],
        output_core_dims=[(Y, X), (Y, X)],
        dask="parallelized",
        vectorize=True,
        output_dtypes=['uint16', bool],
    )
    ds['labels'] = indiv
    ds['peak_mask'] = seeds


def fast_otsu(image, nbins=256, eps=0.1):
    """
    A thin wrapper around skimage.filter.threshold otsu that uses
    fast_histogram.histogram1d to make things ~5x faster per image.

    Parameters
    ----------
    image : np.ndarrary (M,N)
        Grayscale image from which to compute the threshold.
    nbins : int default 265
        Number of bins to compute in the histogram.
    eps : float default = 0.1
        Small offset to expand the edges of the histogram by so
        that the minimum-valued elements get appropriately counted.

    Returns
    -------
    threshold : float
        Threshold value for image. Pixels greater than threshold are considered foreground.
    """
    im_min = image.min()
    im_max = image.max()

    counts = histogram1d(image, nbins, (im_min - eps, im_max + eps))

    bin_width = (im_max - im_min) / nbins

    idx = threshold_otsu(nbins=nbins, hist=counts)

    threshold = im_min + bin_width * (idx + 0.5)

    return threshold


def calc_thresholds(segmentation_images, dims=list('STCZYX')):
    """
    Calculate the threshold for each YX frame in segmentation_images
    using the Otsu's methon. This works well for cells with a
    constituituvely, expressed fluorescent marker.

    segmentation_images : xr.DataArray
        Image dataset to be thresholded. Must have dimensions Y and X
        as named in the dims argument.
    dims: list of str
        Names for dims in segmentation images that correspond to Y and X.
        All other dimensions will be looped over.

    Returns
    -------
    thresh : xr.DataArray
        Thresh will be a dataarray with shape and dims matching
        segmentation_images but with Y X dims dropped.
        Compute the mask with segmentation_images>thresh.
    """
    S, T, C, Z, Y, X = dims

    return xr.apply_ufunc(
        fast_otsu,
        segmentation_images,
        input_core_dims=[[Y, X]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
    )


def relabel_product(labels_arr, fluo, check_labels, min_distance=3):
    """
    Relabel an labels array using the product of fluo and the distance transform
    as the topology for the watershed.

    Parameters
    ----------
    labels_arr : np.array of int
        Array containing labelled regions
    fluo : np.array
        Array containing fluorescnence intensities
    check_labels : np.array of int
        Specific labels in labels_arr which will be updates.
    min_distance : int default 3
        Minimum distance between local maxima to be used as watershed seeds.

    Returns
    -------
    new_labels : np.array of int
        Updated array of labelled regions.
    """

    use_labels = check_labels[check_labels > 0]
    bad_mask = remove_small_holes(np.isin(labels_arr, use_labels))
    dt = ndi.distance_transform_edt(bad_mask)
    norm_fluo = (fluo - fluo.min()) / (fluo.max() - fluo.min())

    seed_pts = peak_local_max(dt * norm_fluo, min_distance=5)
    seed_mask = np.zeros_like(bad_mask)
    seed_mask[tuple(seed_pts.T)] = 1
    seeds_arr, seed_count = ndi.label(seed_mask)

    fixed_bad = watershed(-dt * norm_fluo, markers=seeds_arr)

    new_labels = labels_arr.copy()
    new_labels[bad_mask] = fixed_bad[bad_mask] + labels_arr.max()
    new_labels = relabel_sequential(new_labels)[0]
    return new_labels


def relabel_fluo(mask, fluo, thresh, min_distance=3):
    """
    Apply the watershed transformation to the masked region with
    fluoresence as the topology.

    Parameters
    ----------
    mask : np.array of bool
        Mask of the region that should be relabelled.
    fluo : np.array
        Fluorescence or similar image to to be inverted and used as watershed topology.
    thresh : float
        Threshold for fluorescence images. Pixels below this value will not be
        assigned a label by watershed.
    min_distance : int default 3
        Minimum distance between local maxima to be used when finding watershed seeds.

    Returns
    -------
    indiv : np.array of int
        Labels array.

    """
    peak_idx = peak_local_max(fluo, min_distance=min_distance, threshold_abs=thresh)
    peak_mask = np.zeros_like(mask)
    peak_mask[tuple(peak_idx.T)] = 1
    peak_mask = peak_mask * mask
    labelled_peaks = label(peak_mask)
    indiv = watershed(-fluo, labelled_peaks, mask=mask)
    return indiv


# TODO also rewrite to take labels_arr, bad_labels
def relabel_dt(mask, hit_or_miss_size=5):
    """
    Resegment a labelled region with the hit or miss transform
    and a distance transform based watershed.

    Parameters
    ----------
    mask : np.array of int or bool
        Mask of a single region that needs to be relabeled
    hit_or_miss_size : int default 5
        Size of a square structuring element to be used in
        the hit or miss transform.

    Returns
    -------
    mask : np.array of bool
        Array with mask which may be different from the input
        mask depending on the results of the hit or miss transform
    new_labels: np.array of int with shape of mask
        Array with mask values assigned to new cell identities.

    """
    structure = np.ones((hit_or_miss_size, hit_or_miss_size))
    hom = ndi.binary_hit_or_miss(mask, structure)
    labels, N_pieces = ndi.label(hom, structure=np.ones((3, 3)))
    # If hit or miss generates multiple objects, assume they are cells and
    # use the centroid of each individual as a new seed
    new_seeds_arr = np.zeros(mask.shape)  # zeros_like will return a dask array if mask is dask
    if N_pieces == 0:
        return hom, new_seeds_arr
    elif N_pieces == 1:
        edt = ndi.distance_transform_edt(mask)
        seeds = peak_local_max(edt, min_distance=hit_or_miss_size)
    else:
        edt = ndi.distance_transform_edt(mask)
        coms = ndi.center_of_mass(hom, labels=labels, index=range(1, N_pieces + 1))
        seeds = np.array(coms).round().astype(int)

    new_seeds_arr[tuple(seeds.T)] = 1
    new_seeds_arr = ndi.label(new_seeds_arr)[0]
    relabelled = watershed(-edt, new_seeds_arr, mask=mask, connectivity=2)

    if mask.sum() != (relabelled > 0).sum():
        # Sometimes a few pixels that are in mask do not
        # get assigned to a cell in the watershed procedure
        # return a new mask if that happens to enable boolean indexing
        new_mask = relabelled > 0
        return new_mask, relabelled
    else:
        return mask, relabelled


def merge_overlaps_sequential(prev, curr, overlap_thresh=0.75, area_thresh=200):
    """
    Merge undersegmented cells based on overlaps between cells in successive frames

    prev : np.array of int
        Array containing labelled regions for the earlier time point.
    curr : np.array of int
        Array containing labelled regions for the later time point.
    overlap_thresh : float default 0.75
        Fraction of the area of a cell in prev that must be contained in a cell in
        curr to be elligible to merge.
    area_thresh : int default 200
        Maximum area of a cell that is allowed to be created in the merging process.

    Returns
    -------
    new_prev : np.array of int
        Updated labels array for the earlier time point (i.e. prev)
    """

    overlaps = overlap(prev, curr)

    (overlaps > 0).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        frac_overlaps = overlaps / overlaps.sum(1, keepdims=True)
        # frac_overlaps_ij = fraction of prev_cell_i contained in curr_cell_j

    candidates = frac_overlaps > overlap_thresh
    # multiple cells from prev are essentially contained in a current cell
    merges = candidates.sum(0) > 1
    new_prev = prev.copy()
    current_max = new_prev.max()

    merge_from, merge_into = np.nonzero(merges & candidates)
    for m in np.unique(merge_into):
        old = merge_from[merge_into == m]
        if overlaps[old, m].sum() < area_thresh:
            current_max += 1
            for n in old:
                new_prev[new_prev == n] = current_max
    new_prev = relabel_sequential(new_prev)[0]
    return new_prev


def merge_overlaps_timeseries(labels):
    """
    Merge undersegmented cells through a time series by applying merge_overlaps_sequential
    to each pair of frames starting with the end.

    Parameters
    ----------
    labels : np.array of int with ndim=3 corresponding to T, Y, X
        Time series of labelled regions.

    Returns
    -------
    new_labels : np.array of int
        Updated label array.
    """

    new_labels = labels.copy()
    for i in range(1, labels.shape[0]):
        curr = new_labels[-i]
        prev = new_labels[-i - 1]
        new_labels[-i - 1] = merge_overlaps_sequential(prev, curr)
    return new_labels


def merge_overlaps(labels, dims=list('STCZYX')):
    """
    Merge cells based on overlaps between time points for all labels in dataset

    Parameters
    ----------
    labels : xr.DataArray with at least T, Y, and X dimensions.
        Array with labelled regions.
    dims : list[str]
        Dimension names for the standard 6 dimensions for microscopy datasets.

    Returns
    -------
    new_labels : xr.DataArray same shape/dims/coords as labels
        Updated labels after merging.
    """
    S, T, C, Z, Y, X = dims

    return xr.apply_ufunc(
        merge_overlaps_timeseries,
        labels,
        input_core_dims=[[T, Y, X]],
        output_core_dims=[[T, Y, X]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=['uint16'],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


def labels_to_zarr(labels, out_path):
    """
    Persist a label array as a "sparse" zarr on disk.

    Parameters
    ----------
    labels : np.ndarray
        Labelled images in a numpy array
    out_path : str or Path
        Destination on disk to save the persistent zarr array

    Returns
    -------
    label_zarr: zarr.Array
        Persistent zarr array holding the same data as labels
    """

    coords = labels.nonzero()
    label_vals = labels[coords]
    label_zarr = zarr.open_array(
        out_path, shape=labels.shape, dtype=labels.dtype, chunks=(1, 10, -1, -1)
    )
    label_zarr.set_coordinate_selection(coords, label_vals)
    return label_zarr
