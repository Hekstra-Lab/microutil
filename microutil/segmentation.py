__all__ = [
    "apply_unet",
    "threshold_predictions",
    "individualize",
    "watershed_single_frame_preseeded",
    "peak_mask_to_napari_points",
    "fast_otsu",
    "norm_label2rgb",
    "relabel_bad_cells",
]


import numpy as np
import scipy.ndimage as ndi
import xarray as xr
from fast_histogram import histogram1d
import warnings
import dask.array as da

from skimage.color import label2rgb
from skimage.exposure import equalize_adapthist
from skimage.feature import peak_local_max
from skimage.filters import threshold_isodata, threshold_otsu
from skimage.morphology import label
from skimage.segmentation import relabel_sequential, watershed

from .single_cell import regionprops_df

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


def individualize(ds, topology=None, min_distance=3, connectivity=2, min_area=25, threshold=None):
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
    if isinstance(topology, str):
        use_topology = ds[topology]
        topo_core_dims = list('YX')
    elif isinstance(topology, xr.DataArray):
        use_topology = topology
        topo_core_dims = list('YX')
    else:
        use_topology = None
        topo_core_dims = []

    if topology is not None and threshold is None:

        raise ValueError(
            "Must supply a threshold arry which matches non-core dims of topology."
            "Consider using mu.segmention.fast_otsu on -topology or simply"
            "topology.max(['Y','X']). Failure to pass an array here causes dask"
            "to hang for reasons that I do not fully understand."
        )
    if threshold is not None:
        all_dask = (
            isinstance(ds.mask.data, da.Array)
            and isinstance(topology.data, da.Array)
            and isinstance(threshold.data, da.Array)
        )
        all_numpy = (
            isinstance(ds.mask.data, np.ndarray)
            and isinstance(topology.data, np.ndarray)
            and isinstance(threshold.data, np.ndarray)
        )

    if topology is not None and not (all_dask or all_numpy):
        raise TypeError(
            "ds.mask, topology, and thresh must all be the same type of array (dask or numpy) but found a mix"
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
        input_core_dims=[["Y", "X"], topo_core_dims, []],
        output_core_dims=[("Y", "X"), ("Y", "X")],
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
    """
    im_min = image.min()
    im_max = image.max()

    counts = histogram1d(image, nbins, (im_min - eps, im_max + eps))

    bin_width = (im_max - im_min) / nbins

    idx = threshold_otsu(nbins=nbins, hist=counts)

    threshold = im_min + bin_width * (idx + 0.5)

    return threshold


def norm_label2rgb(labels):
    """
    Lightly wrap skimage.colors.label2rgb to return floats in [0,1)
    rather than ints for display in matplotlib without warnings.

    Parameters
    ----------
    labels : np.array of int
        Array containing labelled regions with background having value 0.

    Returns
    -------
    show_labels: np.array of float with shape=labels.shape+(3,)
    """
    show = label2rgb(labels, bg_label=0)
    c_max = show.max((0, 1))
    return show / c_max


def make_show_labels(
    ds, label_name='labels', show_label_name='show_labels', rgb_dim_name='rgb', dims=list('STCZYX')
):
    """
    Add a new variable to ds that contains label regions colored with norm_label2rb.

    Parameters
    ----------
    ds: xr.Datset
        Datset containing labelled images
    label_name: str default "labels"
        Name of labelled image variable in ds.
    show_label_name: str default "show_labels"
        Name of the new variable that will contain the colored label images.
    rgb_dim_name: str default 'rgb'
        Name of the dim that will contain the color channels.
    dims: str or list of str default list('STCZYX')
        Names of standard dims in the dataset. Color will be assigned
        to labels in each YX frame independently

    Returns
    -------
    Nothing. ds is updated inplace.
    """

    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    elif isinstance(dims, list):
        S, T, C, Z, Y, X = dims

    ds[show_label_name] = xr.apply_ufunc(
        norm_label2rgb,
        ds[label_name],
        input_core_dims=[[Y, X]],
        output_core_dims=[[Y, X, rgb_dim_name]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={'output_sizes': {rgb_dim_name: 3}},
    )


def _get_bbox_and_label(i, df, pad=5, max_sizes_yx=(1024, 1020)):
    label = df.loc[i, 'label'].astype(int)
    ymin, xmin, ymax, xmax = df.loc[i, [f"bbox-{j}" for j in range(4)]].values.astype(int)

    ymin = ymin - pad if ymin - pad > 0 else 0
    xmin = xmin - pad if xmin - pad > 0 else 0
    ymax = ymax + pad if ymax + pad <= max_sizes_yx[0] else max_sizes_yx[0] + 1
    xmax = xmax + pad if xmax + pad <= max_sizes_yx[1] else max_sizes_yx[1] + 1

    return label, ymin, ymax, xmin, xmax


def relabel_dt(mask, hit_or_miss_size=5):
    """
    Resegment a labelled region with the hit or miss transform
    and a distance transform based watershed.

    Parameters
    ----------
    mask: np.array of int or bool
        Mask of a single region that needs to be relabeled
    hit_or_miss_size: int default 5
        Size of a square structuring element to be used in
        the hit or miss transform.

    Returns
    -------
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
        return new_seeds_arr

    elif N_pieces == 1:
        edt = ndi.distance_transform_edt(mask)
        seeds = peak_local_max(edt, min_distance=hit_or_miss_size)
    else:
        edt = ndi.distance_transform_edt(mask)
        coms = ndi.center_of_mass(hom, labels=labels, index=range(1, N_pieces + 1))
        seeds = np.array(coms).round().astype(int)

    new_seeds_arr[tuple(seeds.T)] = 1
    new_seeds_arr = ndi.label(new_seeds_arr)[0]
    return watershed(-edt, new_seeds_arr, mask=mask, connectivity=2)

def get_neighbor_bbox_and_mask(labels_arr, lookup_mask, structure_size=3, pad=5):
    structure = np.ones((structure_size, structure_size))
    edges = ndi.binary_dilation(lookup_mask, structure)&(~lookup_mask)
    neighbors = np.unique(labels_arr[edges])
    neighbors = neighbors[neighbors>0]

    out = lookup_mask
    for n in neighbors:
        out |= labels_arr==n

    if out.sum()==0:
        bbox = None
    else:
        bbox = regionprops_df(out.astype('uint8'), ['bbox'], {}).values.squeeze()

    return bbox, out


def relabel_fluo(mask, fluo, thresh, min_distance=3):
    peak_idx = peak_local_max(fluo, min_distance=min_distance, threshold_abs=thresh)
    peak_mask = np.zeros_like(mask)
    peak_mask[tuple(peak_idx.T)] = 1
    labelled_peaks = label(peak_mask)
    indiv = watershed(-fluo, labelled_peaks, mask=mask)
#    indiv = relabel_sequential(indiv)[0]
    return indiv


def relabel_fluo_framewise(frame_labels, check_labels, frame_fluo, frame_thresh):
    current_max = frame_labels.max()
    new_labels = frame_labels.copy()
    use_labels = check_labels[check_labels>0]
    for i, lab in enumerate(use_labels):
        if lab not in new_labels:
            continue  # We have already updated that label

        lookup_mask = new_labels==lab
        bbox, neighbor_mask = get_neighbor_bbox_and_mask(new_labels, lookup_mask)
        if bbox is None:
            continue  # this cell has been merged out of existence already
        ymin, xmin, ymax, xmax = bbox
        bbox_fluo = frame_fluo[ymin:ymax, xmin:xmax]
        # need to resize the bbox to fit all the neighbors
        mask = neighbor_mask[ymin:ymax, xmin:xmax]
        new_bbox_labels = relabel_fluo(mask, bbox_fluo, frame_thresh)

        if lookup_mask.sum() != (new_bbox_labels > 0).sum():
            # Sometimes a few pixels that are in mask do not
            # get assigned to a cell in the watershed procedure
            # zero them out in frame_labels to remove them from the
            # dataset and update frame_label_mask so boolean indexing
            # assignment still works below.
            new_labels[lookup_mask] = 0
            lookup_mask = np.zeros_like(lookup_mask)
            mask = new_bbox_labels > 0
            lookup_mask[ymin:ymax, xmin:xmax] = mask > 0

        cell_count = new_bbox_labels.max()
        new_labels[lookup_mask] = new_bbox_labels[mask] + current_max
        current_max += cell_count
    return new_labels

def relabel_hybrid(frame_labels, frame_fluo, frame_thresh, check_labels, check_areas, area_low, area_high,  pad_size=3):
    current_max = frame_labels.max()
    structure = np.ones((pad_size, pad_size))
    new_labels = frame_labels.copy()
    use_labels = check_labels[check_labels>0]
    use_areas = check_areas[check_labels>0]
    for i, lab in enumerate(use_labels):
        if lab not in new_labels:
            continue  # We have already updated that label
            
        lookup_mask = new_labels==lab
        edges = ndi.binary_dilation(lookup_mask, structure)&(~lookup_mask)
        neighbors, overlaps =  np.unique(new_labels[edges], return_counts=True)
        not_bkgd = neighbors>0
        neighbors = neighbors[not_bkgd]
        if len(neighbors)==0:
            continue  # nobody to merge with
        overlaps = overlaps[not_bkgd]
        bad_neighbors = list(set(use_labels) & set(neighbors))
        if len(bad_neighbors)==0:
            if use_areas[i]<area_low:
                # if its small force merge it 
                max_overlap = np.argmax(overlaps)
                new_labels[lookup_mask] = neighbors[max_overlap]
            elif use_areas[i]>area_high:
                # if its large use fluo and dont consider neighbors
                mask = lookup_mask.astype('uint8')
                bbox = regionprops_df(mask, ['bbox'],{}) 
                ymin, xmin, ymax, xmax = bbox.values.squeeze()
                bbox_mask = mask[ymin:ymax, xmin:xmax].astype(bool)
                bbox_fluo = frame_fluo[ymin:ymax, xmin:xmax]
                relabelled = relabel_fluo(bbox_mask, bbox_fluo, frame_thresh)
                cell_count = np.max(relabelled)
                new_labels[mask.astype(bool)] = relabelled[bbox_mask]+current_max
                current_max += cell_count

        elif len(bad_neighbors)<3:
            max_overlap = np.argmax([overlaps[neighbors==bn] for bn in bad_neighbors])
            new_labels[lookup_mask] = bad_neighbors[max_overlap] 
        else: #multiple bad neighbors
            mask = lookup_mask
            for n in bad_neighbors:
                mask |= new_labels==n
            mask = mask.astype('uint8')
            bbox = regionprops_df(mask, ['bbox'],{}) 
            ymin, xmin, ymax, xmax = bbox.values.squeeze()
            bbox_mask = mask[ymin:ymax, xmin:xmax].astype(bool)
            relabelled = relabel_dt(bbox_mask)
            cell_count = np.max(relabelled)
            new_labels[mask.astype(bool)] = relabelled[bbox_mask]+current_max
            current_max += cell_count
    new_labels = relabel_sequential(new_labels)[0]
    return new_labels

def relabel_bad_cells(
    ds,
    cell_props_df,
    method,
    label_name='labels',
    dims=list('STCZYX'),
    threshold=None,
    fluo=None,
    hit_or_miss_size=5,
):
    """
    Change ds.labels in place to automatically correct poorly segmented cells.
    ***Note that currently this does not update the seeds**

    Parameters
    ----------
    ds: xr.Dataset
        Dataset containing the cell labels
    """

    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    elif isinstance(dims, list):
        S, T, C, Z, Y, X = dims

    if not isinstance(ds[label_name].data, np.ndarray):
        ds.labels.load()

    current_max = ds.labels.max([Y, X]).data

    if method == "dt":
        relabel = relabel_dt
    elif method == "fluo":
        relabel = relabel_fluo
        if not (isinstance(fluo, xr.DataArray) and (Y in fluo.dims) and (X in fluo.dims)):
            raise ValueError("Attempting to use method=\"fluo\" but found invalid fluo kwarg")

        if threshold is None:
            warnings.warn(
                "No threshold value passed. Its recommended to get a threshold value\
                           from applying fast_otsu to each frame in fluorescent images"
            )
        fluo = fluo.load()
        threshold = threshold.load()

    loop_sizes = {k: v for k, v in ds.labels.sizes.items() if k not in [Y, X]}

    frame_groupby = cell_props_df.groupby(list(loop_sizes.keys()))

    for dims, frame_cell_props in frame_groupby:
        frame_labels = ds.labels.data[dims]
        if method == "fluo":
            frame_fluo = fluo.data[dims]
            frame_thresh = threshold.data[dims]

        # TODO this inner loop could be wrapped and dask.delayed to do this in parallel
        # That would be advantageous for doing this iteratively since each iteration 
        # could be done with dask rather than just the first.
        # this will be a fxn that takes (label_arr, bad_labels) and return new_labels.
        # That thing could be wrapped in xr.apply_ufunc as long as the set of bad labels
        # Gets stored in a padded dataarray.
        for idx, row in frame_cell_props.iterrows():
            lab = row['label']
            frame_label_mask = frame_labels == lab
            if method == "fluo":
                bbox, neighbor_mask = get_neighbor_bbox_and_mask(frame_labels, frame_label_mask)
                if bbox is None:
                    continue  # this cell has been merged out of existence already
                ymin, xmin, ymax, xmax = bbox
                bbox_fluo = frame_fluo[ymin:ymax, xmin:xmax]
                # need to resize the bbox to fit all the neighbors
                mask = neighbor_mask[ymin:ymax, xmin:xmax]
                new_labels = relabel(mask, bbox_fluo, frame_thresh)
            else:
                _, ymin, ymax, xmin, xmax = _get_bbox_and_label(idx, frame_cell_props)
                mask = frame_label_mask[ymin:ymax, xmin:xmax]
                new_labels = relabel(mask)
            if frame_label_mask.sum() != (new_labels > 0).sum():
                # Sometimes a few pixels that are in mask do not
                # get assigned to a cell in the watershed procedure
                # zero them out in frame_labels to remove them from the
                # dataset and update frame_label_mask so boolean indexing
                # assignment still works below.
                frame_labels[frame_label_mask] = 0
                frame_label_mask = np.zeros_like(frame_label_mask)
                frame_label_mask[ymin:ymax, xmin:xmax] = new_labels > 0

                mask = new_labels > 0
            cell_count = new_labels.max()
            frame_labels[frame_label_mask] = new_labels[mask] + current_max[dims]
            current_max[dims] += cell_count
        ds.labels.data[dims] = relabel_sequential(frame_labels)[0]
