__all__ = [
    "construct_cost_matrix",
    "frame_to_features",
    "track_single_pos",
    "track",
]

import scipy
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import numpy as np
import xarray as xr
from .track_utils import _reindex_labels, reindex


def norm(prev, curr):
    for col in range(3):
        r = np.hstack([prev[:, col], curr[:, col]])
        m = np.mean(r)
        s = np.std(r)
        prev[:, col] = (prev[:, col] - m) / s
        curr[:, col] = (curr[:, col] - m) / s


def frame_to_features(frame):
    """
    Parameters
    ----------
    frame : (X, Y) array-like
        The mask of labels to create features from

    Returns
    -------
    features : (N, 3)
        The features array of (com_x, com_y, area)
    """
    indexed_frame, labels, areas = _reindex_labels(frame, None)
    com = np.asarray(ndi.center_of_mass(indexed_frame, indexed_frame, labels[1:]))
    return np.hstack([com, areas[1:, None]])


def construct_cost_matrix(
    prev,
    curr,
    weights=[1, 1, 1 / 20],
    pad=1e4,
    debug_info='',
    normalize=False,
    distance_cutoff=0.5,
    compute_overlap=True,
):
    """
    prev : (X, Y) array of int
        The previous time point's labels
    curr : (X, Y) array of int
        The current time point's labels.
    weights : (3,) array-like, default: [1, 1, 1/5]
        The weighting of features to use for the minkowski distance.
        The current order is [X, Y, area]
    pad : number or None, default: 1e4
        The value to use when padding the cost matrix to be square. Set to *None*
        to not pad.
    normalize : bool, default: False
        Whether to normalize the each frames features. Optional as it sometimes seems
        to cause issues.
    distance_cutoff : float, default: .5
        Float between [0,1] the maximum distance relative to the frame size
        for which cells can be considered to be tracked. Cell pairs with a distance
        geater than the computed maximum will be given an entry into the cost matrix of
        1e6
    compute_overlap : bool, default: True
        Whether to to weight the assignments by how much the cells overlap between the
        two timesteps. This may be a slow step.

    Returns
    -------
    C : (N, N) array
        The cost matrix. Where *N* is the larger of the number of cells of the two
        time points
    M : int
        The number of cells in the previous time point.
    """
    prev_features = frame_to_features(prev)
    curr_features = frame_to_features(curr)
    xy_dist = scipy.spatial.distance.cdist(prev_features[:, :2], curr_features[:, :2])
    if normalize:
        norm(prev_features, curr_features)

    C = scipy.spatial.distance.cdist(prev_features, curr_features, metric="minkowski", w=weights)

    max_dist = np.sqrt(prev.shape[0] ** 2 + prev.shape[1] ** 2)
    too_far_idx = xy_dist > distance_cutoff * max_dist
    C[too_far_idx] = 1e6

    # figure out if masks overlap and make those ones more likely
    def unpadded_overlap(prev, curr, shape):
        p_uniq = np.unique(prev)
        c_uniq = np.unique(curr)
        arr = np.zeros(shape)
        for i in p_uniq:
            for j in c_uniq:
                arr[i - 1, j - 1] = np.sum((prev == i) * (curr == j))
        return arr

    if compute_overlap:
        overlaps = unpadded_overlap(prev, curr, C.shape)
        overlaps /= overlaps.max()
        C *= 1 - overlaps

    if np.any(np.isnan(C)):
        print(prev_features)
        print(curr_features)
        print(C)
    M, N = C.shape
    if pad is not None:
        if M < N:
            # maybe these should be low cost connections?
            row_pad = N - M
            C = np.pad(C, ((0, row_pad), (0, 0)), constant_values=pad)
        elif M > N:
            print('oh boi!')
            print(debug_info + f' - {M} {N}')
        return C, M
    return C, M


def track_single_pos(cells, weights=[1, 1, 1 / 5], pad=1e4):
    """
    Attempt to keep cells' labels the same over time points.

    Parameters
    ----------
    cells : (T, M, N) array-like
        The mask of cell labels. Should be integers.
    weights : (3,) array-like, default: [1, 1, 1/5]
        The weighting of features to use for the minkowski distance.
        The current order is [X, Y, area]
    pad : number or None, default: 1e4
        The value to use when padding the cost matrix to be square. Set to *None*
        to not pad.

    Returns
    -------
    tracked : (T, M, N) array
        The labels mask with cells tracked through time. So ideally
        a cell labelled 4 at t0 will also be labelled 4 at t1.
    """
    # Ineffecient to call _reindex_labels multiple times
    # but currently just trying to get it working so not going to restructure
    # plus unique is pretty performant so the hit isn't so big
    # we can't keep the areas from this because they are not per time point
    arr = reindex(cells, inplace=False)
    tracked = np.zeros_like(arr)
    tracked[0] = arr[0]

    # for loop for now.
    # xarray rolling operations look promising, but I couldn't get them working.
    # TODO: double and flip the graph to allow for lineage tracking
    # https://www.hpl.hp.com/techreports/2012/HPL-2012-40R1.pdf

    for t in range(1, len(cells)):
        C, M = construct_cost_matrix(tracked[t - 1], arr[t], weights=weights, debug_info=f't={t}')
        row_ind, col_ind = linear_sum_assignment(C)
        assignments = np.stack([row_ind, col_ind], axis=1)

        for i in range(len(assignments)):
            prev, curr = assignments[i]
            idx = arr[t] == curr + 1
            tracked[t][idx] = prev + 1
    return tracked


def track(
    ds, weights=[1, 1, 1 / 5], pad=1e4, normalize=False, distance_cutoff=0.5, compute_overlap=True
):
    """
    Attempt to keep cells' labels the same over time points. This will modify
    the *labels* variable of the dataset in place

    Parameters
    ----------
    ds : (S, T, ..., Y, X) Dataset
        The dataset to use. Should contain a variable *labels*
    weights : (3,) array-like, default: [1, 1, 1/5]
        The weighting of features to use for the minkowski distance.
        The current order is [X, Y, area]
    pad : number or None, default: 1e4
        The value to use when padding the cost matrix to be square. Set to *None*
        to not pad.
    normalize : bool, default: False
        Whether to normalize the each frames features. Optional as it sometimes seems
        to cause issues.
    distance_cutoff : float, default: .5
        Float between [0,1] the maximum distance relative to the frame size
        for which cells can be considered to be tracked. Cell pairs with a distance
        geater than the computed maximum will be given an entry into the cost matrix of
        1e6
    compute_overlap : bool, default: True
        Whether to to weight the assignments by how much the cells overlap between the
        two timesteps. This may be a slow step.
    """
    # for loop for now.
    # xarray rolling operations look promising, but I couldn't get them working.
    # TODO: double and flip the graph to allow for lineage tracking
    # https://www.hpl.hp.com/techreports/2012/HPL-2012-40R1.pdf
    from skimage.segmentation import relabel_sequential

    def f(arr):
        return relabel_sequential(arr)[0]

    # for some reason apply_ufunc wasn't work here
    for s in range(ds.dims['S']):
        for t in range(ds.dims['T']):
            ds['labels'][s, t] = relabel_sequential(ds['labels'][s, t].values)[0]

    for s in range(ds.dims['S']):
        labels = ds['labels'][s].values
        arr = np.copy(ds['labels'][s].values)
        for t in range(1, ds.dims['T']):
            C, M = construct_cost_matrix(
                labels[t - 1],
                labels[t],
                weights=weights,
                debug_info=f'{s=}, t={t}',
                normalize=normalize,
                distance_cutoff=distance_cutoff,
                compute_overlap=compute_overlap,
            )
            row_ind, col_ind = linear_sum_assignment(C)
            assignments = np.stack([row_ind, col_ind], axis=1)

            for i in range(len(assignments)):
                prev, curr = assignments[i]
                idx = arr[t] == curr + 1
                ds['labels'][s][t].values[idx] = prev + 1
