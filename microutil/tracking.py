__all__ = [
    "construct_cost_matrix",
    "frame_to_features",
    "track",
]

import scipy
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import numpy as np
from .track_utils import _reindex_labels


def _normalize(arr):
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    return (arr - mean) / std


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
    indexed_frame, labels, areas = _reindex_labels(frame)
    com = np.asarray(ndi.center_of_mass(indexed_frame, indexed_frame, labels[1:]))
    return _normalize(np.hstack([com, areas[1:, None]]))


def construct_cost_matrix(prev, curr, weights=[1, 1, 1 / 5], pad=1e4):
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

    C = scipy.spatial.distance.cdist(prev_features, curr_features, metric="minkowski", w=weights)
    # dists = scipy.spatial.distance.cdist(prev_features[:,:2], curr_features[:,:2])
    # C[dists>200] = 1e9
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
            C[-row_pad:, N:]
        elif M > N:
            print('oh boi!')
            print(M, N)
        return C, M
    return C, M


def track(cells, weights=[1, 1, 1 / 5], pad=1e4):
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
    tracked = np.zeros_like(cells)
    tracked[0] = cells[0]

    # for loop for now.
    # xarray rolling operations look promising, but I couldn't get them working.
    # TODO: double and flip the graph to allow for lineage tracking
    # https://www.hpl.hp.com/techreports/2012/HPL-2012-40R1.pdf
    for t in range(1, len(cells)):
        C, M = construct_cost_matrix(tracked[t - 1], cells[t], weights=weights)
        row_ind, col_ind = linear_sum_assignment(C)
        assignments = np.stack([row_ind, col_ind], axis=1)

        for i in range(len(assignments)):
            prev, curr = assignments[i]
            idx = cells[t] == curr + 1
            if i >= M:
                tracked[t][idx] = curr + 1
            else:
                tracked[t][idx] = prev + 1
    return tracked
